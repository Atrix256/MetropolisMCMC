#include <stdio.h>
#include <math.h>
#include <random>
#include <vector>

inline float Lerp(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

inline float Clamp(float v, float min, float max)
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
        // TODO: this isn't quite right
        float percent = (value - m_min) / (m_max - m_min);
        size_t bucket = size_t(0.5f + percent * float(m_numBuckets));
        if (bucket >= m_numBuckets - 1)
            bucket = m_numBuckets - 1;
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
    // are we drawing random numbers or getting expected value (integrating?). Maybe both? i dunno...

    Histogram histogram(xmin, xmax, 100);

    FILE* file = nullptr;
    fopen_s(&file, "out/out.csv", "w+t");

    std::random_device rd;
    std::seed_seq fullSeed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
    std::mt19937 rng(fullSeed);

    std::normal_distribution<float> normalDist(0.0f, stepSizeSigma);
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

    float xcurrent = xstart;
    float ycurrent = function(xstart);
    float expectedValue = ycurrent;

    fprintf(file, "\"index\",\"x\",\"y\",\"expected value\"\n");
    fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\"\n", size_t(0), xcurrent, ycurrent, expectedValue);
    histogram.AddValue(xcurrent);

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
        }

        expectedValue = Lerp(expectedValue, ycurrent, 1.0f / float(sampleIndex));

        fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%f\"\n", sampleIndex, xcurrent, ycurrent, expectedValue);
        histogram.AddValue(xcurrent);
    }

    // TODO: spit out values into a CSV and do a histogram with it?
    fclose(file);

    fopen_s(&file, "out/histogram.csv", "w+t");
    fprintf(file, "\"Bucket Index\",\"Bucket X\",\"Actual Y\",\"Count\",\"Percentage\",\n");

    for (size_t index = 0; index < histogram.m_numBuckets; ++index)
    {
        float percent = float(index) / float(histogram.m_numBuckets - 1);
        float x = xmin + percent * (xmax - xmin);
        float y = function(x);

        fprintf(file, "\"%zu\",\"%f\",\"%f\",\"%zu\",\"%f\",\n", index, x, y, histogram.m_counts[index], float(histogram.m_counts[index])/float(sampleCount));
    }

    fclose(file);
}

float Sin(float x)
{
    return sinf(x);
}


int main(int argc, char** argv)
{
    MetropolisMCMC(Sin, 0.0f, 1.0f, 0.5f, 0.1f, 100000);

    return 0;
}

/*

TODO:

* Clamp to range instead of putting f in function.
* Also make smaller step size! Maybe be based on range?
* average y value is the expected value.
 ? i did this one but it doesn't seem to be correct! (or is it?)

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
 


 Links:
 - real good! https://stephens999.github.io/fiveMinuteStats/MH_intro.html


Attachments area
Preview YouTube video Markov Chain Monte Carlo and the Metropolis Alogorithm

https://youtu.be/h1NOS_wxgGg
https://www.youtube.com/watch?v=3ZmW_7NXVvk
https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50
https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/


Next: read up on metropolis light transport

 */