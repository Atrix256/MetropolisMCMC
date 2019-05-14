#include <stdio.h>
#include <math.h>
#include <random>
#include <vector>
#include <array>

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
void MetropolisMCMC(const FUNCTION& function, float xstart, float stepSizeSigma, size_t sampleCount)
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
    float expectedValue = 0.0f;
    {
        fopen_s(&file, "out/out.csv", "w+t");

        float xcurrent = xstart;
        float ycurrent = function(xstart);
        expectedValue = ycurrent;

        fprintf(file, "\"index\",\"x\",\"y\",\"expected value\"\n");
        fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\"\n", size_t(0), xcurrent, ycurrent, expectedValue);

        ymin = ycurrent;
        ymax = ycurrent;

        samples[0][0] = xcurrent;
        samples[0][1] = ycurrent;

        for (size_t sampleIndex = 1; sampleIndex < sampleCount; ++sampleIndex)
        {
            float xnext = xcurrent + normalDist(rng);
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
    float xmin = samples[0][0];
    float xmax = samples[0][0];
    for (const std::array<float, 2>& s : samples)
    {
        xmin = std::min(xmin, s[0]);
        xmax = std::max(xmax, s[0]);
    }
    Histogram histogram(xmin, xmax, 100);
    for (const std::array<float, 2>& s : samples)
        histogram.AddValue(s[0]);

    float definiteIntegral = expectedValue * (xmax-xmin);

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
    MetropolisMCMC(Sin, 0.5f, 0.2f, 100000);

    //MetropolisMCMC(AbsSin, 0.5f, 0.2f, 100000);

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



Next: read up on metropolis light transport
Next: Hamilton Monte Carlo since you can do big jumps if you can get derivative (dual numbers!)

 */