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

template <size_t N>
class Histogram
{
public:
    Histogram(const TSample<N>& min, const TSample<N>& max, size_t numBucketsPerAxis)
        : m_min(min)
        , m_max(max)
        , m_numBucketsPerAxis(numBucketsPerAxis)
    {
        size_t totalBuckets = 1;
        for (size_t i = 0; i < N; ++i)
            totalBuckets *= m_numBucketsPerAxis;
        m_counts.resize(totalBuckets, 0);
    }

    size_t GetActualBucketIndex(const std::array<size_t, N>& bucketIndex) const
    {
        size_t actualBucket = 0;
        for (int i = N - 1; i >= 0; --i)
        {
            actualBucket *= m_numBucketsPerAxis;
            actualBucket += bucketIndex[i];
        }
        return actualBucket;
    }

    void AddValue(const TSample<N>& value)
    {
        std::array<size_t, N> bucketIndex;
        for (int i = 0; i < N; ++i)
        {
            float percent = (value[i] - m_min[i]) / (m_max[i] - m_min[i]);
            bucketIndex[i] = Clamp<size_t>(size_t(percent * float(m_numBucketsPerAxis)), 0, m_numBucketsPerAxis - 1);
        }

        m_counts[GetActualBucketIndex(bucketIndex)]++;
    }

    // Get a bucket's [min, max)
    void GetBucketMinMax(const std::array<size_t, N>& index, TSample<N>& min, TSample<N>& max) const
    {
        for (int i = 0; i < N; ++i)
        {
            min[i] = Lerp(m_min[i], m_max[i], float(index[i]) / float(m_numBucketsPerAxis));
            max[i] = Lerp(m_min[i], m_max[i], float(index[i] + 1) / float(m_numBucketsPerAxis));
        }
    }

    // a helper to iterate through the buckets of an N dimensional histogram
    bool IterateBucketIndices(std::array<size_t, N>& indices) const
    {
        for (size_t i = 0; i < N; ++i)
        {
            indices[i]++;
            if (indices[i] != m_numBucketsPerAxis)
                return true;

            indices[i] = 0;
        }

        return false;
    }

    size_t GetBucketCount(const std::array<size_t, N> indices) const
    {
        return m_counts[GetActualBucketIndex(indices)];
    }

    TSample<N> m_min;
    TSample<N> m_max;
    size_t m_numBucketsPerAxis;
    std::vector<size_t> m_counts;
};

std::random_device g_rd;
std::seed_seq g_fullSeed{ g_rd(), g_rd(), g_rd(), g_rd(), g_rd(), g_rd(), g_rd(), g_rd() };
std::mt19937 g_rng(g_fullSeed);

template <typename FUNCTION, size_t N>
float CalculateNormalizationConstant(const FUNCTION& function, const Histogram<N>& histogram, size_t sampleCount, size_t sampleCountNormalizationConstant)
{
    // Find largest count histogram bucket.
    // Calculate it as a percentage of total samples, call this C.
    std::array<size_t, N> indices;
    std::fill(indices.begin(), indices.end(), 0);
    std::array<size_t, N> largestCountIndex = indices;
    size_t largestCount = histogram.GetBucketCount(largestCountIndex);
    do
    {
        size_t count = histogram.GetBucketCount(indices);
        if (count > largestCount)
        {
            largestCountIndex = indices;
            largestCount = count;
        }
    }
    while (histogram.IterateBucketIndices(indices));
    float C = float(histogram.m_counts.size()) * float(largestCount) / float(sampleCount);

    // Integrate the function numerically over that largest bucket, using white noise
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
    float D = 0.0f;
    {
        TSample<N> sampleMin, sampleMax;
        histogram.GetBucketMinMax(largestCountIndex, sampleMin, sampleMax);
        for (size_t index = 0; index < sampleCountNormalizationConstant; ++index)
        {
            // make it be from sampleMin to sampleMax, so it's in the histogram bucket
            TSample<N> sample;
            for (int i = 0; i < N; ++i)
                sample[i] = Lerp(sampleMin[i], sampleMax[i], uniformDist(g_rng));

            D = Lerp(D, function( sample ), 1.0f / float(index + 1));
        }
        for (int i = 0; i < N; ++i)
            D *= (histogram.m_max[i] - histogram.m_min[i]);
    }

    // the estimate of the normalization constant is D / C
    // D is the integration of the un normalized pdf for that one histogram bucket.
    // C is the integration of the normalized pdf for that one histogram bucket.
    return D / C;
}

template <typename FUNCTION, size_t N>
void MetropolisMCMC(const FUNCTION& function, const TSample<N-1>& start, float stepSizeSigma, size_t sampleCount, std::vector<TSample<N>>& samples)
{
    // run our mcmc code to generate sample points
    samples.resize(sampleCount);

    std::normal_distribution<float> normalDist(0.0f, stepSizeSigma);
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

    // make the starting sample
    TSample<N-1> currentSample = start;
    float currentValue = std::max(function(currentSample), 0.0f);

    // store the first sample, as the place we start at
    for (int i = 0; i < N-1; ++i)
        samples[0][i] = currentSample[i];
    samples[0][N - 1] = currentValue;

    // do a random walk
    for (size_t sampleIndex = 1; sampleIndex < sampleCount; ++sampleIndex)
    {
        // get a proposed next sample for our random walk
        TSample<N - 1> nextSample;
        for (int i = 0; i < N - 1; ++i)
            nextSample[i] = currentSample[i] + normalDist(g_rng);
        float nextValue = std::max(function(nextSample), 0.0f);

        // take the new sample if ynext > ycurrent (it's more probable), or with a probability based on how much less probable it is.
        float A = nextValue / currentValue;
        if (A >= 1.0f || uniformDist(g_rng) < A)
        {
            currentSample = nextSample;
            currentValue = nextValue;
        }

        // store the current sample, whether we took the new sample or not
        for (int i = 0; i < N - 1; ++i)
            samples[sampleIndex][i] = currentSample[i];
        samples[sampleIndex][N - 1] = currentValue;
    }
}

template <typename FUNCTION, size_t N>
void Report(
    const FUNCTION& function,
    size_t histogramBucketCount,
    size_t integrationHistogramBucketCount,
    size_t sampleCountNormalizationConstant,
    std::vector<TSample<N>>& samples,
    const char* samplesFileName,
    const char* histogramFileName
)
{
    // calculate the min and max of the samples
    TSample<N-1> sampleMin, sampleMax;
    for (int i = 0; i < N - 1; ++i)
        sampleMin[i] = sampleMax[i] = samples[0][i];
    for (const TSample<N>& s : samples)
    {
        for (int i = 0; i < N-1; ++i)
        {
            sampleMin[i] = std::min(sampleMin[i], s[i]);
            sampleMax[i] = std::max(sampleMax[i], s[i]);
        }
    }

    // calculate the normalization constant so we can estimate the integral
    // Use a different histogram with fewer larger buckets to make the MCMC part of the estimate more accurate
    size_t sampleCount = samples.size();
    Histogram<N-1> histogram2(sampleMin, sampleMax, integrationHistogramBucketCount);
    for (const TSample<N>& s : samples)
    {
        TSample<N - 1> s2;
        for (int i = 0; i < N - 1; ++i)
            s2[i] = s[i];
        histogram2.AddValue(s2);
    }
    float normalizationConstant = CalculateNormalizationConstant(function, histogram2, sampleCount, sampleCountNormalizationConstant);

    // Make a histogram of the function input to show that the samples follow the shape of the function.
    // In other words, show that the function was used as a PDF, which described the probabilities of each possible value.
    Histogram<N-1> histogram(sampleMin, sampleMax, histogramBucketCount);
    for (const std::array<float, N>& s : samples)
    {
        TSample<N - 1> s2;
        for (int i = 0; i < N - 1; ++i)
            s2[i] = s[i];
        histogram.AddValue(s2);
    }

    // Write out the sample data, while also calculating the expected value at each step.
    // The final value of expected value should be taken as the most accurate.
    TSample<N - 1> expectedValue;
    for (float& f : expectedValue)
        f = 0.0f;
    {
        FILE* file = nullptr;
        fopen_s(&file, samplesFileName, "w+t");
        fprintf(file, "\"index\"");
        for (int i = 0; i < N; ++i)
            fprintf(file, ",\"sample[%i]\"", i);
        for (int i = 0; i < N-1; ++i)
            fprintf(file, ",\"expectedValue[%i]\"", i);
        fprintf(file, "\n");

        for (size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
        {
            for (int i = 0; i < N-1; ++i)
                expectedValue[i] = Lerp(expectedValue[i], samples[sampleIndex][i], 1.0f / float(sampleIndex + 1));  // incrementally averaging: https://blog.demofox.org/2016/08/23/incremental-averaging/

            fprintf(file, "\"%zu\"", sampleIndex);
            for (int i = 0; i < N; ++i)
                fprintf(file, ",\"%f\"", samples[sampleIndex][i]);
            for (int i = 0; i < N-1; ++i)
                fprintf(file, ",\"%f\"", expectedValue[i]);
            fprintf(file, "\n");
        }

        fclose(file);
    }

    // write out a histogram file
    {
        FILE* file = nullptr;
        fopen_s(&file, histogramFileName, "w+t");
        fprintf(file, "\"Bucket Index 0\"");
        for (int i = 1; i < N - 1; ++i)
        {
            fprintf(file, ",\"Bucket Index %i\"", i);
        }
        for (int i = 0; i < N - 1; ++i)
            fprintf(file, ",\"Bucket Value %i\"", i);
        fprintf(file, ",\"Function Value\",\"Normalized Function Value\",\"Count\",\"Percentage\",\n");

        // find out what the normalization constant is of the real function for the histogram buckets
        float normalizationConstant = 0.0f;
        std::array<size_t, N - 1> indices;
        std::fill(indices.begin(), indices.end(), 0);
        do
        {
            TSample<N - 1> functionInput;
            for (int i = 0; i < N - 1; ++i)
            {
                float percent = (float(indices[i]) + 0.5f) / float(histogram.m_numBucketsPerAxis);
                functionInput[i] = sampleMin[i] + percent * (sampleMax[i] - sampleMin[i]);
            }
            float functionValue = std::max(function(functionInput), 0.0f);
            normalizationConstant += functionValue;

        }
        while (histogram.IterateBucketIndices(indices));

        // print out the histogram data
        std::fill(indices.begin(), indices.end(), 0);
        do
        {
            TSample<N - 1> functionInput;
            for (int i = 0; i < N - 1; ++i)
            {
                float percent = (float(indices[i]) + 0.5f) / float(histogram.m_numBucketsPerAxis);
                functionInput[i] = sampleMin[i] + percent * (sampleMax[i] - sampleMin[i]);
            }
            float functionValue = std::max(function(functionInput), 0.0f);

            fprintf(file, "\"%zu\"", indices[0]);
            for (int i = 1; i < N - 1; ++i)
                fprintf(file, ",\"%zu\"", indices[i]);
            for (int i = 0; i < N - 1; ++i)
                fprintf(file, ",\"%f\"", functionInput[i]);

            fprintf(file, ",\"%f\",\"%f\",\"%zu\",\"%f\",\n", functionValue, functionValue / normalizationConstant, histogram.GetBucketCount(indices), float(histogram.GetBucketCount(indices)) / float(sampleCount));
        }
        while (histogram.IterateBucketIndices(indices));

        fclose(file);
    }

    // show results
    printf("%zu samples taken\nIntegration = %f\nexpected value = ", sampleCount, normalizationConstant);
    for (int i = 0; i < N - 1; ++i)
    {
        if (i == 0)
            printf("%f", expectedValue[i]);
        else
            printf(", %f", expectedValue[i]);
    }
    printf("\n\n");
}

// integrating sin(x) from 0 to pi == 2
float Sin(const std::array<float, 1>& x)
{
    if (x[0] < 0.0f || x[0] > c_pi)
        return 0.0f;

    return std::max(sinf(x[0]), 0.0f);
}

// integrating sin(x)*sin(x) from 0 to 2 pi == pi
float SinSquared(const std::array<float, 1>& x)
{
    if (x[0] < 0.0f || x[0] > 2.0f * c_pi)
        return 0.0f;

    return std::max(sinf(x[0])*sinf(x[0]), 0.0f);
}

// integrating |sin(x)| from 0 to 2 pi == 4
float AbsSin(const std::array<float, 1>& x)
{
    if (x[0] < 0.0f || x[0] > 2.0f * c_pi)
        return 0.0f;

    return fabsf(sinf(x[0]));
}

// integrating xyy from x in [0,2] y in [0,1] = 2/3
// https://mathinsight.org/double_integral_examples
float XYSquared(const std::array<float, 2>& xy)
{
    if (xy[0] < 0.0f || xy[0] > 2.0f || xy[1] < 0.0f || xy[1] > 1.0f)
        return 0.0f;

    return std::max(xy[0] * xy[1] * xy[1], 0.0f);
}

int main(int argc, char** argv)
{
    // y = sin(x) from 0 to pi
    {
        printf("y=sin(x)   x in [0,pi]\n");
        std::vector<std::array<float, 2>> samples;
        MetropolisMCMC(Sin, { c_pi / 2.0f }, 0.2f, 1000000, samples);
        Report(Sin, 100, 10, 1000000, samples, "out/samples_Sin.csv", "out/histogram_Sin.csv");
    }

    // y = sin(x) * sin(x) from 0 to 2 pi
    {
        printf("y=sin(x)*sin(x)   x in [0,2pi]\n");
        std::vector<std::array<float, 2>> samples;
        MetropolisMCMC(SinSquared, { c_pi / 2.0f }, 0.2f, 1000000, samples);
        Report(SinSquared, 100, 10, 1000000, samples, "out/samples_SinSq.csv", "out/histogram_SinSq.csv");
    }

    // y = |sin(x)| from 0 to 2 pi
    {
        printf("y=|sin(x)|   x in [0,2pi]\n");
        std::vector<std::array<float, 2>> samples;
        MetropolisMCMC(AbsSin, { c_pi / 2.0f }, 0.2f, 1000000, samples);
        Report(AbsSin, 100, 10, 1000000, samples, "out/samples_AbsSin.csv", "out/histogram_AbsSin.csv");
    }

    // z = xyy where x in [0,2] and y in [0,1]
    {
        printf("z=xyy  x in [0,2] y in [0,1]\n");
        std::vector<std::array<float, 3>> samples;
        MetropolisMCMC(XYSquared, { 1.0f, 0.5f }, 0.2f, 1000000, samples);
        Report(XYSquared, 100, 10, 1000000, samples, "out/samples_xysquared.csv", "out/histogram_xysquared.csv");
    }

    system("Pause");

    return 0;
}

/*

 * excel can do 3d histograms. can open office?

 


 Notes:
* if function goes negative, I don't think it ever chooses probability values there.
 * yeah. it makes the probability of taking the new point be effectively zero, because the if case can never be true.
* I think clamping biases things. maybe makes it not symetric? i dunno.
 * yeah. it moves it to the end. It should just make it not move at all. That's why having the function return 0 out of range works.
* you have to tune how fast you move x around.
 * could imagine making it smaller over time. simulated annealing style. cooling rate another hyper parameter though.

 * integrating using white noise. not the best at all but it works in any dimension.

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
  * this true, at least for sin(x) from 0 to pi. 100000 MCMC samples. 100 histogram buckets. 0.2 sigma and initial guess of 1.57. using sobol...
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