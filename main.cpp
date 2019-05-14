#include <stdio.h>
#include <math.h>
#include <random>
#include <vector>
#include <array>
#include <conio.h>

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

    float m_min;
    float m_max;
    size_t m_numBuckets;
    std::vector<size_t> m_counts;
};

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
            if (uniformDist(rng) < A)
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

    // Write out the sample data, while also calculating the expected value and integral at each step.
    // The final values of integral and expected value should be taken as the most accurate.
    float integral = 0.0f;
    float expectedValue = 0.0f;
    {
        FILE* file = nullptr;
        fopen_s(&file, "out/samples.csv", "w+t");
        fprintf(file, "\"index\",\"x\",\"y\",\"expected value\",\"integral\"\n");

        for (size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
        {
            float value = samples[sampleIndex][1] / (xmax - xmin); // f(x) / p(x)
            integral = Lerp(integral, value, 1.0f / float(sampleIndex + 1)); // incrementally averaging: https://blog.demofox.org/2016/08/23/incremental-averaging/

            expectedValue = Lerp(expectedValue, samples[sampleIndex][0], 1.0f / float(sampleIndex + 1));  // more incremental averaging

            fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\",\"%f\"\n", sampleIndex, samples[sampleIndex][0], samples[sampleIndex][1], expectedValue, integral);
        }

        fclose(file);
    }

    // Make a histogram of the x axis to show that they follow the shape of the function.
    // That is, the function was used as a PDF, which described the probabilities of each possible value.
    Histogram histogram(xmin, xmax, histogramBucketCount);
    for (const std::array<float, 2>& s : samples)
        histogram.AddValue(s[0]);

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
    printf("expected value = %0.2f\nIntegral = %0.2f from %0.2f to %0.2f\n", expectedValue, integral, xmin, xmax);
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
    MetropolisMCMC(Sin, c_pi / 2.0f, 0.2f, 100000, 100);

    system("Pause");

    return 0;
}

/*

TODO:

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


Attachments area
Preview YouTube video Markov Chain Monte Carlo and the Metropolis Alogorithm

https://youtu.be/h1NOS_wxgGg
https://www.youtube.com/watch?v=3ZmW_7NXVvk
https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50
https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/

https://www.johndcook.com/blog/2016/01/23/introduction-to-mcmc/

https://www.johndcook.com/blog/2016/01/25/mcmc-burn-in/

This code just picks a new point every time. Not quite MCMC, since there is no random walk.
https://bl.ocks.org/eliangcs/6e8b45f88fd3767363e7

Next: read up on metropolis light transport
Next: Hamilton Monte Carlo since you can do big jumps if you can get derivative (dual numbers!)

 */