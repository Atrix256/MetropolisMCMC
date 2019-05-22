#include <stdio.h>
#include <math.h>
#include <random>
#include <vector>
#include <array>
#include <conio.h>

static const float c_pi = 3.14159265359f;

template <size_t N>
using TSample = std::array<float, N>;

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
    { }

    void AddValue(float value)
    {
        float percent = (value - m_min) / (m_max - m_min);
        size_t bucket = Clamp<size_t>(size_t(percent * float(m_numBuckets)), 0, m_numBuckets - 1);
        m_counts[bucket]++;
    }

    // Get a bucket's [min, max)
    void GetBucketMinMax(size_t index, float& min, float& max) const
    {
        min = Lerp(m_min, m_max, float(index) / float(m_numBuckets));
        max = Lerp(m_min, m_max, float(index + 1) / float(m_numBuckets));
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
float CalculateNormalizationConstant(const FUNCTION& function, const Histogram& histogram, size_t sampleCount, size_t sampleCountNormalizationConstant)
{
    // Find largest count histogram bucket.
    // Calculate it as a percentage of total samples, call this C.
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

    // Integrate the function numerically over that largest bucket, using 1d sobol
    float D = 0.0f;
    {
        float sampleMin, sampleMax;
        histogram.GetBucketMinMax(largestCountIndex, sampleMin, sampleMax);
        size_t sampleInt = 0;
        for (size_t index = 0; index < sampleCountNormalizationConstant; ++index)
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

    // the estimate of the normalization constant is D / C
    // D is the integration of the un-normalized pdf for that one histogram bucket.
    // C is the integration of the normalized pdf for that one histogram bucket.
    return D / C;
}

template <typename FUNCTION, size_t N>
void MetropolisMCMC(const FUNCTION& function, const TSample<N-1>& start, float stepSizeSigma, size_t sampleCount, std::vector<TSample<N>>& samples)
{
    // run our mcmc code to generate sample points
    samples.resize(sampleCount);
    std::random_device rd;
    std::seed_seq fullSeed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
    std::mt19937 rng(fullSeed);

    std::normal_distribution<float> normalDist(0.0f, stepSizeSigma);
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

    // make the starting sample
    //TSample<N> currentSample = start;
    //float currentValue = std::max(function(currentSample), 0.0f);

    // make the starting sample
    float xcurrent = start[0];
    float ycurrent = std::max(function(xcurrent), 0.0f);

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

template <typename FUNCTION>
void Report(
    const FUNCTION& function,
    size_t histogramBucketCount,
    size_t integrationHistogramBucketCount,
    size_t sampleCountNormalizationConstant,
    std::vector<std::array<float, 2>>& samples,
    const char* samplesFileName,
    const char* histogramFileName
)
{
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

    // calculate the normalization constant so we can estimate the integral
    // Use a different histogram with fewer larger buckets to make the MCMC part of the estimate more accurate
    size_t sampleCount = samples.size();
    Histogram histogram2(xmin, xmax, integrationHistogramBucketCount);
    for (const std::array<float, 2>& s : samples)
        histogram2.AddValue(s[0]);
    float normalizationConstant = CalculateNormalizationConstant(function, histogram2, sampleCount, sampleCountNormalizationConstant);

    // Make a histogram of the x axis to show that the samples follow the shape of the function.
    // In other words, show that the function was used as a PDF, which described the probabilities of each possible value.
    Histogram histogram(xmin, xmax, histogramBucketCount);
    for (const std::array<float, 2>& s : samples)
        histogram.AddValue(s[0]);

    // Write out the sample data, while also calculating the expected value at each step.
    // The final value of expected value should be taken as the most accurate.
    float expectedValue = 0.0f;
    {
        FILE* file = nullptr;
        fopen_s(&file, samplesFileName, "w+t");
        fprintf(file, "\"index\",\"x\",\"y\",\"expected value\"\n");
        for (size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
        {
            expectedValue = Lerp(expectedValue, samples[sampleIndex][0], 1.0f / float(sampleIndex + 1));  // incrementally averaging: https://blog.demofox.org/2016/08/23/incremental-averaging/
            fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\"\n", sampleIndex, samples[sampleIndex][0], samples[sampleIndex][1], expectedValue);
        }

        fclose(file);
    }

    // write out a histogram file
    {
        FILE* file = nullptr;
        fopen_s(&file, histogramFileName, "w+t");
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
    printf("%zu samples taken\nexpected value = %f\nIntegration = %f\n\n", sampleCount, expectedValue, normalizationConstant);
}

// integrating sin(x) from 0 to pi == 2
float Sin(float x)
{
    if (x < 0.0f || x > c_pi)
        return 0.0f;

    return std::max(sinf(x), 0.0f);
}

// integrating sin(x)*sin(x) from 0 to 2 pi == pi
float SinSquared(float x)
{
    if (x < 0.0f || x > 2.0f * c_pi)
        return 0.0f;

    return std::max(sinf(x)*sin(x), 0.0f);
}

// integrating |sin(x)| from 0 to 2 pi == 4
float AbsSin(float x)
{
    if (x < 0.0f || x > 2.0f * c_pi)
        return 0.0f;

    return fabsf(sinf(x));
}

int main(int argc, char** argv)
{
    // y = sin(x) from 0 to pi
    {
        std::vector<std::array<float, 2>> samples;
        MetropolisMCMC(Sin, { c_pi / 2.0f }, 0.2f, 100000, samples);
        Report(Sin, 100, 10, 1000, samples, "out/samples_Sin.csv", "out/histogram_Sin.csv");
    }

    // y = sin(x) * sin(x) from 0 to 2 pi
    {
        std::vector<std::array<float, 2>> samples;
        MetropolisMCMC(SinSquared, { c_pi / 2.0f }, 0.2f, 100000, samples);
        Report(SinSquared, 100, 10, 1000, samples, "out/samples_SinSq.csv", "out/histogram_SinSq.csv");
    }

    // y = |sin(x)| from 0 to 2 pi
    {
        std::vector<std::array<float, 2>> samples;
        MetropolisMCMC(AbsSin, { c_pi / 2.0f }, 0.2f, 100000, samples);
        Report(AbsSin, 100, 10, 1000, samples, "out/samples_AbsSin.csv", "out/histogram_AbsSin.csv");
    }

    system("Pause");

    return 0;
}

/*

TODO:

? 2d example?
 * have mcmc and reporting use TSample so we can do 2d next

- use this for sampling from a function as if it were a PDF.
 - maybe a couple 1d examples, and maybe a 2d example?
 - could use STB to show data. or just use excel.

 * excel can do 3d histograms

 


 Notes:
* if function goes negative, I don't think it ever chooses probability values there.
 * yeah. it makes the probability of taking the new point be effectively zero, because the if case can never be true.
* I think clamping biases things. maybe makes it not symetric? i dunno.
 * yeah. it moves it to the end. It should just make it not move at all. That's why having the function return 0 out of range works.
* you have to tune how fast you move x around.
 * could imagine making it smaller over time. simulated annealing style. cooling rate another hyper parameter though.

 * random walk doesn't have to be gaussian but it commonly is.

 * samples aren't independent, so convergance rates not as well known

* Burn in / get to 0.234 acceptance rate by tuning sigma
 * didn't play with step size a whole lot
 * tried briefly to auto-tune sigma but couldn't reliably get it to be close & not infinite loop

* show the value of a good initial guess, by showing convergence with a good vs bad guess.
 * i guess it depends on step size too...

- this is continuous PDF, but works for discrete PMF too.
 - i think the main differences are...
 - 1) when moving to next state, choose randomly from neighbors
 - 2) Also need a possibility for staying in the same state (else, you can't always possibly be at every state each step). Unsure specific probabilities
 - Metropolis algorithm is simpler due to being able to assume symetry. Metropolis - Hastings allows asymetry. read docs for more info.


 * when estimating the normalization constant
  * 1000 samples of sobol integration (D) vs the "highest count bucket" (C).  C often had 3-8 times as much error as D, and is 1/2 the value, so has even higher percentage error.
  * this true, at least for sin(x) from 0 to pi. 100000 MCMC samples. 100 histogram buckets. 0.2 sigma and initial guess of 1.57
  * taking it to 10 buckets, it was consistently about 0.03 off, while the sobol integration was 0.06. making them even by percentage error.
  * this was before the auto tuning of guess and sigma

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


Link for sobol, to sample zoo:
https://github.com/Atrix256/SampleZoo


Odd note: this is kinda integration with red noise. compare to blue?  Makes you wonder how the rejection stuff would play out with blue noise or LDS
 * possibly hybrid? do some number of MCMC and some white noise.

? how would you use MCMC for searching a sorted list? plant the seed.

Integration Methods:
1) the worst Monte Carlo method
2) use tricky math to make normalization constant cancel out (#1 is an example)
3) your histogram bucket Monte Carlo idea. Find largest bucket. Monte Carlo integrate over same range. Divide for estimate of normalization constant.

Next: read up on metropolis light transport
Next: Hamilton Monte Carlo since you can do big jumps if you can get derivative (dual numbers!)
Next: check out gibbs sampling?


 */