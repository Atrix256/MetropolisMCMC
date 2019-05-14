#include <stdio.h>
#include <math.h>
#include <random>
#include <vector>
#include <array>

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
void MetropolisMCMC(const FUNCTION& function, float xmin, float xmax, float xstart, float stepSizeSigma, size_t sampleCount)
{
    std::random_device rd;
    std::seed_seq fullSeed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
    std::mt19937 rng(fullSeed);

    std::normal_distribution<float> normalDist(0.0f, stepSizeSigma);
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

    FILE* file = nullptr;

    // run our mcmc code, generate and write out the data
    std::vector<std::array<float, 2>> samples(sampleCount);
    float ymin = 0.0f;
    float ymax = 0.0f;
    {
        fopen_s(&file, "out/out.csv", "w+t");

        float xcurrent = xstart;
        float ycurrent = function(xstart);
        float expectedValue = ycurrent;

        fprintf(file, "\"index\",\"x\",\"y\",\"expected value\"\n");
        fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\"\n", size_t(0), xcurrent, ycurrent, expectedValue);

        ymin = ycurrent;
        ymax = ycurrent;

        samples[0][0] = xcurrent;
        samples[0][1] = ycurrent;

        for (size_t sampleIndex = 1; sampleIndex < sampleCount; ++sampleIndex)
        {
            float xnext = Clamp(xcurrent + normalDist(rng), xmin, xmax);
            float ynext = function(xnext);

            // take the new x if y next > ycurrent, or with some probability based on how much worse it is.
            float A = ynext / ycurrent;
            if (uniformDist(rng) < A)
            {
                xcurrent = xnext;
                ycurrent = ynext;

                if (ycurrent < ymin)
                    ymin = ycurrent;

                if (ycurrent > ymax)
                    ymax = ycurrent;
            }

            expectedValue = Lerp(expectedValue, ycurrent, 1.0f / float(sampleIndex + 1));

            samples[sampleIndex][0] = xcurrent;
            samples[sampleIndex][1] = ycurrent;

            fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\"\n", sampleIndex, xcurrent, ycurrent, expectedValue);
        }
        fclose(file);
    }

    // make a histogram
    Histogram histogram(xmin, xmax, 100);
    for (const std::array<float, 2>& s : samples)
        histogram.AddValue(s[0]);

    // write out a histogram file
    {
        fopen_s(&file, "out/histogram.csv", "w+t");
        fprintf(file, "\"Bucket Index\",\"Bucket X\",\"Actual Y\",\"Normalized Y\",\"Count\",\"Percentage\",\n");

        // find out what the normalization constant is of the real function for the histogram buckets
        float normalizationConstant = 0.0f;
        for (size_t index = 0; index < histogram.m_numBuckets; ++index)
        {
            float percent = (float(index) + 0.5f) / float(histogram.m_numBuckets - 1);
            float x = xmin + percent * (xmax - xmin);
            float y = function(x);
            normalizationConstant += y;
        }

        // print out the histogram data
        for (size_t index = 0; index < histogram.m_numBuckets; ++index)
        {
            float percent = (float(index) + 0.5f) / float(histogram.m_numBuckets - 1);
            float x = xmin + percent * (xmax - xmin);
            float y = function(x);

            fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\",\"%zu\",\"%f\",\n", index, x, y, y / normalizationConstant, histogram.m_counts[index], float(histogram.m_counts[index]) / float(sampleCount));
        }

        fclose(file);
    }
}

float Sin(float x)
{
    return sinf(x);
}

float IndefiniteIntegral_Sin(float x)
{
    return -cosf(x);
}

int main(int argc, char** argv)
{

    MetropolisMCMC(Sin, 0.0f, 3.5f, 0.5f, 0.1f, 100000);

    return 0;
}

/*

TODO:

* i think clamping X biases things.
 * I think instead, you should have it just return zero outside of where you want to go? try it and see.


* Also make smaller step size! Maybe be based on range?
 * need to play around with step sizes

* average y value is the expected value.
 ? i did this one but it doesn't seem to be correct! (or is it?)
 ? is expected value the x or the y in this case?

* in the histogram, the actual calculated y's sum up to BUCKET_COUNT * 0.45970 (integration of sin(x) from 0 to 1).
 * the counts sum up to sample count, which can be normalized to a percentage, but if we are looking for a mean, what do we do?
 * maybe take the mean x and plug it in to get the mean y?
 ? we could also make our function a PDF by integrating it and dividing by that as a normalization constant. I don't like that though... you have to know too much about the function.


* you have to tune how fast you move x around.
 * could imagine making it smaller over time. simulated annealing style. cooling rate another hyper parameter though.
 * could do a line search.  Probably should? talk about it in notes on blog i guess.

* show both usage cases?
 ? you can use this to find the expected value (mean) which lets you integrate
 ? you can also use this to draw random numbers

? how fast is convergance?

- use this for sampling from a function as if it were a PDF.
 - maybe a couple 1d examples, and maybe a 2d example?
 - could use STB to show data.

- this is continuous PDF, but works for discrete PMF too.
 - i think the main differences are...
 - 1) when moving to next state, choose randomly from neighbors
 - 2) Also need a possibility for staying in the same state (else, you can't always possibly be at every state each step). Unsure specific probabilities

 - Metropolis algorithm is simpler due to being able to assume symetry. Metropolis - Hastings allows asymetry. read docs for more info.
 


 Notes:
 * if function goes negative, I don't think it ever chooses probability values there.  Experiment and understand.
 * I think clamping biases things. maybe makes it not symetric? i dunno.



 Links:
 - real good! https://stephens999.github.io/fiveMinuteStats/MH_intro.html


Attachments area
Preview YouTube video Markov Chain Monte Carlo and the Metropolis Alogorithm

https://youtu.be/h1NOS_wxgGg
https://www.youtube.com/watch?v=3ZmW_7NXVvk
https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50
https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/



Next: read up on metropolis light transport
Next: Hamilton Monte Carlo since you can do big jumps if you can get derivative (dual numbers!)

 */