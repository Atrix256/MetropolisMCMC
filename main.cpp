#include <stdio.h>
#include <math.h>
#include <random>
#include <vector>
#include <array>
#include <conio.h>

static const size_t c_numSamplesNormalizationConstant = 1000;

static const float c_pi = 3.14159265359f;

inline float Lerp(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

template<typename T>
inline T Clamp(T v, T min, T max)
{
    if (v <= min)
        return min;
    else if (v >= max)
        return max;
    else
        return v;
}

class Histogram
{
public:
    Histogram(float min, float max, size_t numBuckets)
        : m_min(min)
        , m_max(max)
        , m_numBuckets(numBuckets)
        , m_counts(numBuckets, 0)
    {

    }

    void AddValue(float value)
    {
        float percent = (value - m_min) / (m_max - m_min);
        size_t bucket = Clamp<size_t>(size_t(percent * float(m_numBuckets)), 0, m_numBuckets - 1);
        m_counts[bucket]++;
    }

    // [min, max)
    void GetBucketMinMax(size_t index, float& min, float& max) const
    {
        min = (float(index) / float(m_numBuckets)) * (m_max - m_min) + m_min;
        max = float(index + 1) / float(m_numBuckets) * (m_max - m_min) + m_min;
    }

    float m_min;
    float m_max;
    size_t m_numBuckets;
    std::vector<size_t> m_counts;
};

static size_t Ruler(size_t n)
{
    size_t ret = 0;
    while (n != 0 && (n & 1) == 0)
    {
        n /= 2;
        ++ret;
    }
    return ret;
}

void Sobol(std::vector<float>& values, size_t numValues)
{
    values.resize(numValues);
    size_t sampleInt = 0;
    for (size_t i = 0; i < numValues; ++i)
    {
        size_t ruler = Ruler(i + 1);
        size_t direction = size_t(size_t(1) << size_t(31 - ruler));
        sampleInt = sampleInt ^ direction;
        values[i] = float(sampleInt) / std::pow(2.0f, 32.0f);
    }
}

template <typename FUNCTION>
float CalculateNormalizationConstant(const FUNCTION& function, const Histogram& histogram, size_t sampleCount)
{
    // Find largest count histogram bucket.
    // Calculate it as a sample percentage, call this C.
    size_t largestCount = histogram.m_counts[0];
    size_t largestCountIndex = 0;

    for (size_t index = 1; index < histogram.m_counts.size(); ++index)
    {
        if (histogram.m_counts[index] > largestCount)
        {
            largestCountIndex = index;
            largestCount = histogram.m_counts[index];
        }
    }
    float C = float(histogram.m_counts.size()) * float(largestCount) / float(sampleCount);

    // Integrate the function numerically over the largest bucket, using 1d sobol
    float D = 0.0f;
    {
        float sampleMin, sampleMax;
        histogram.GetBucketMinMax(largestCountIndex, sampleMin, sampleMax);
        size_t sampleInt = 0;
        for (size_t index = 0; index < c_numSamplesNormalizationConstant; ++index)
        {
            // get the sobol sample from 0 to 1
            size_t ruler = Ruler(index + 1);
            size_t direction = size_t(size_t(1) << size_t(31 - ruler));
            sampleInt = sampleInt ^ direction;
            float sample = float(sampleInt) / std::pow(2.0f, 32.0f);

            // make it be from sampleMin to sampleMax, so it's in the histogram bucket
            sample = Lerp(sampleMin, sampleMax, sample);

            D = Lerp(D, function(sample), 1.0f / float(index + 1));
        }
        D *= (histogram.m_max - histogram.m_min);
    }

    // the estimate of the normalization constant is D/C
    return (D / C);
}

template <typename FUNCTION>
void MetropolisMCMC(const FUNCTION& function, float xstart, float stepSizeSigma, size_t sampleCount, size_t histogramBucketCount)
{
    // run our mcmc code to generate sample points
    std::vector<std::array<float, 2>> samples(sampleCount);
    {
        std::random_device rd;
        std::seed_seq fullSeed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
        std::mt19937 rng(fullSeed);

        std::normal_distribution<float> normalDist(0.0f, stepSizeSigma);
        std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

        // make the starting sample
        float xcurrent = xstart;
        float ycurrent = std::max(function(xstart), 0.0f);

        // store the first sample, as the place we start at
        samples[0][0] = xcurrent;
        samples[0][1] = ycurrent;

        // do a random walk
        for (size_t sampleIndex = 1; sampleIndex < sampleCount; ++sampleIndex)
        {
            // get a proposed next sample for our random walk
            float xnext = xcurrent + normalDist(rng);
            float ynext = std::max(function(xnext), 0.0f);

            // take the new sample if ynext > ycurrent (it's more probable), or with a probability based on how much less probable it is.
            float A = ynext / ycurrent;
            if (A >= 1.0f || uniformDist(rng) < A)
            {
                xcurrent = xnext;
                ycurrent = ynext;
            }

            // store the current sample, whether we took the new sample or not
            samples[sampleIndex][0] = xcurrent;
            samples[sampleIndex][1] = ycurrent;
        }
    }

    // calculate the min and max x and y of the samples
    float xmin = samples[0][0];
    float xmax = samples[0][0];
    float ymin = samples[0][1];
    float ymax = samples[0][1];
    for (const std::array<float, 2>& s : samples)
    {
        xmin = std::min(xmin, s[0]);
        xmax = std::max(xmax, s[0]);
        ymin = std::min(ymin, s[1]);
        ymax = std::max(ymax, s[1]);
    }

    // Make a histogram of the x axis to show that the samples follow the shape of the function.
    // In other words, show that the function was used as a PDF, which described the probabilities of each possible value.
    Histogram histogram(xmin, xmax, histogramBucketCount);
    for (const std::array<float, 2>& s : samples)
        histogram.AddValue(s[0]);

    // calculate the normalization constant so we can estimate the integral
    float normalizationConstant = CalculateNormalizationConstant(function, histogram, sampleCount);

    // Write out the sample data, while also calculating the expected value at each step.
    // The final value of expected value should be taken as the most accurate.
    float expectedValue = 0.0f;
    {
        FILE* file = nullptr;
        fopen_s(&file, "out/samples.csv", "w+t");
        fprintf(file, "\"index\",\"x\",\"y\",\"expected value\"\n");

        float integral = 0.0f;
        for (size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
        {
            expectedValue = Lerp(expectedValue, samples[sampleIndex][0], 1.0f / float(sampleIndex + 1));  // incrementally averaging: https://blog.demofox.org/2016/08/23/incremental-averaging/

            float pdf = samples[sampleIndex][1] / normalizationConstant;

            // TODO: this integral calculation is wrong, remove it
            float estimate = samples[sampleIndex][0] / pdf; // f(x) / p(x)
            integral = Lerp(integral, estimate, 1.0f / float(sampleIndex + 1));

            fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\"\n", sampleIndex, samples[sampleIndex][0], samples[sampleIndex][1], expectedValue);
        }

        fclose(file);
    }

    // write out a histogram file
    {
        FILE* file = nullptr;
        fopen_s(&file, "out/histogram.csv", "w+t");
        fprintf(file, "\"Bucket Index\",\"Bucket X\",\"Actual Y\",\"Normalized Y\",\"Count\",\"Percentage\",\n");

        // find out what the normalization constant is of the real function for the histogram buckets
        float normalizationConstant = 0.0f;
        for (size_t index = 0; index < histogram.m_numBuckets; ++index)
        {
            float percent = (float(index) + 0.5f) / float(histogram.m_numBuckets - 1);
            float x = xmin + percent * (xmax - xmin);
            float y = std::max(function(x), 0.0f);
            normalizationConstant += y;
        }

        // print out the histogram data
        for (size_t index = 0; index < histogram.m_numBuckets; ++index)
        {
            float percent = (float(index) + 0.5f) / float(histogram.m_numBuckets - 1);
            float x = xmin + percent * (xmax - xmin);
            float y = std::max(function(x), 0.0f);

            fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\",\"%zu\",\"%f\",\n", index, x, y, y / normalizationConstant, histogram.m_counts[index], float(histogram.m_counts[index]) / float(sampleCount));
        }

        fclose(file);
    }

    // show results
    printf("%zu samples taken\nexpected value = %0.2f\n", sampleCount, expectedValue);
}

float Sin(float x)
{
    if (x < 0.0f || x > c_pi)
        return 0.0f;

    return std::max(sinf(x), 0.0f);
}

float SinSquared(float x)
{
    if (x < 0.0f || x > 2.0f * c_pi)
        return 0.0f;

    return std::max(sinf(x)*sin(x), 0.0f);
}

float AbsSin(float x)
{
    if (x < 0.0f || x > 2.0f * c_pi)
        return 0.0f;

    return fabsf(sinf(x));
}

int main(int argc, char** argv)
{
    MetropolisMCMC(Sin, c_pi / 2.0f, 0.2f, 100000, 100); // TODO: more histogram buckets!

    system("Pause");

    return 0;
}

/*

TODO:

* The normalization constant IS the integral. Find a way to report that as it makes it.

* Burn in / get to 0.234 acceptance rate by tuning sigma

* do the thing about using the histogram to estimate the normalization constant
 * it'd be nice to show it estimating over time.


* try this for your examples...
https://twitter.com/Reedbeta/status/1129598841747935232


* this seems helpful:
http://www.pmean.com/07/MetropolisAlgorithm.html

* your integral is 0.2 instead of 2.0. Why??
 * also i can't understand why this works... the random walk is pretty deterministic.  Maybe is it because the p(x) needs to account for taking a gaussian step?

* make a wrapper function that will clamp a function by returning 0 when out of bounds.
 * this is for integrating functions between specific values

* actually use this to integrate. the expected value is not quite there.

? does the random walk have to be gaussian? could it be white noise?
 * i think so, but which is better?

* Also make smaller step size! Maybe be based on range?
 * need to play around with step sizes

* average y value is the expected value.
 ? i did this one but it doesn't seem to be correct! (or is it?)
 ? is expected value the x or the y in this case?

 ? 2d example?



* show both usage cases?
 ? you can use this to find the expected value (mean) which lets you integrate
 ? you can also use this to draw random numbers

? how fast is convergance?

- use this for sampling from a function as if it were a PDF.
 - maybe a couple 1d examples, and maybe a 2d example?
 - could use STB to show data. or just use excel.


 


 Notes:
* if function goes negative, I don't think it ever chooses probability values there.
 * yeah. it makes the probability of taking the new point be effectively zero, because the if case can never be true.
* I think clamping biases things. maybe makes it not symetric? i dunno.
 * yeah. it moves it to the end. It should just make it not move at all. That's why having the function return 0 out of range works.
* you have to tune how fast you move x around.
 * could imagine making it smaller over time. simulated annealing style. cooling rate another hyper parameter though.

* show the value of a good initial guess, by showing convergence with a good vs bad guess.
 * i guess it depends on step size too...

- this is continuous PDF, but works for discrete PMF too.
 - i think the main differences are...
 - 1) when moving to next state, choose randomly from neighbors
 - 2) Also need a possibility for staying in the same state (else, you can't always possibly be at every state each step). Unsure specific probabilities
 - Metropolis algorithm is simpler due to being able to assume symetry. Metropolis - Hastings allows asymetry. read docs for more info.


 Links:
 - real good! https://stephens999.github.io/fiveMinuteStats/MH_intro.html

 http://www.pmean.com/07/MetropolisAlgorithm.html


https://youtu.be/h1NOS_wxgGg
https://www.youtube.com/watch?v=3ZmW_7NXVvk
https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50
https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/

https://www.johndcook.com/blog/2016/01/23/introduction-to-mcmc/

https://www.johndcook.com/blog/2016/01/25/mcmc-burn-in/

https://ermongroup.github.io/cs323-notes/probabilistic/mh/

This code just picks a new point every time. Not quite MCMC, since there is no random walk.
https://bl.ocks.org/eliangcs/6e8b45f88fd3767363e7

I guess getting an integral of f(x) is not possible?
https://stats.stackexchange.com/questions/248652/approximating-1d-integral-with-metropolis-hastings-markov-chain-monte-carlo

monte carlo:
https://blog.demofox.org/2018/06/12/monte-carlo-integration-explanation-in-1d/
https://theclevermachine.wordpress.com/tag/monte-carlo-integration/

https://radfordneal.wordpress.com/2008/08/17/the-harmonic-mean-of-the-likelihood-worst-monte-carlo-method-ever/

good twitter thread
https://twitter.com/Atrix256/status/1129545765603479558


Odd note: this is kinda integration with red noise. compare to blue?  Makes you wonder how the rejection stuff would play out with blue noise or LDS


? how would you use MCMC for searching a sorted list? plant the seed.

Integration Methods:
1) the worst Monte Carlo method
2) use tricky math to make normalization constant cancel out (#1 is an example)
3) your histogram bucket Monte Carlo idea. Find largest bucket. Monte Carlo integrate over same range. Divide for estimate of normalization constant.

Next: read up on metropolis light transport
Next: Hamilton Monte Carlo since you can do big jumps if you can get derivative (dual numbers!)
Next: check out gibbs sampling?


 */