package algoscale

import (
	"context"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"
)

// 💥 1. Stress Test Memori Concurrent / Stress Under High Load
func TestPenetration_1_ConcurrentStress(t *testing.T) {
	const goroutines = 50
	const iterations = 10

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				// Test LogSinkhornStreaming
				m, n := 8, 8
				logR := make([]float64, m)
				logC := make([]float64, n)
				for k := range logR {
					logR[k] = -math.Log(float64(m))
				}
				for k := range logC {
					logC[k] = -math.Log(float64(n))
				}
				costFn := func(r, c int) float64 { return float64((r - c) * (r - c)) }
				cfg := LogSinkhornConfig{
					Epsilon:       0.1,
					Tau1:          1.0,
					Tau2:          1.0,
					MaxIterations: 20,
					Tolerance:     1e-4,
				}
				_, err1 := LogSinkhornStreaming(m, n, logR, logC, costFn, cfg)
				if err1 != nil {
					t.Errorf("Concurrent LogSinkhorn failed: %v", err1)
				}

				// Test ComputeBarycenterLogDomain
				dists := [][]float64{{0.5, 0.5}, {0.2, 0.8}}
				weights := []float64{0.5, 0.5}
				baryCfg := BarycenterConfig{
					Epsilon:       0.5,
					MaxIterations: 20,
					Tolerance:     1e-4,
				}
				_, err2 := ComputeBarycenterLogDomain(dists, weights, func(a, b int) float64 { return math.Abs(float64(a - b)) }, 2, 2, baryCfg)
				if err2 != nil {
					t.Errorf("Concurrent Barycenter failed: %v", err2)
				}
			}
		}(g)
	}

	wg.Wait()
}

// 🌊 2. Subnormal / Denormal Number Bomb (10^-308 ~ 10^-323)
func TestPenetration_2_SubnormalNumberBomb(t *testing.T) {
	subnormal := math.SmallestNonzeroFloat64 // ~5e-324

	vals := []float64{subnormal, subnormal * 2, subnormal * 10}
	resLSE := LogSumExp(vals)
	if math.IsNaN(resLSE) || math.IsInf(resLSE, 0) {
		t.Fatalf("LogSumExp failed on subnormal numbers: %f", resLSE)
	}

	m, n := 3, 3
	logR := []float64{math.Log(subnormal), -1.0, -1.0}
	logC := []float64{-1.0, math.Log(subnormal), -1.0}

	costFn := func(i, j int) float64 {
		return subnormal * float64(i+j+1)
	}

	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 20,
		Tolerance:     1e-4,
	}

	res, err := LogSinkhornStreaming(m, n, logR, logC, costFn, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming failed on subnormals: %v", err)
	}
	if math.IsNaN(res.Cost) {
		t.Fatalf("Cost is NaN with subnormals")
	}
}

// 🕳️ 3. Floating-Point Negative Zero (-0.0) & Sign Bit Inversion
func TestPenetration_3_NegativeZeroInjection(t *testing.T) {
	negZero := math.Copysign(0.0, -1)

	if math.Signbit(negZero) != true {
		t.Fatalf("Failed to construct negative zero")
	}

	costFn := func(i, j int) float64 {
		if i == j {
			return negZero
		}
		return 1.0
	}

	m, n := 2, 2
	logR := []float64{-0.693, -0.693}
	logC := []float64{-0.693, -0.693}

	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 20,
		Tolerance:     1e-4,
	}

	res, err := LogSinkhornStreaming(m, n, logR, logC, costFn, cfg)
	if err != nil {
		t.Fatalf("Failed under negative zero injection: %v", err)
	}
	if math.IsNaN(res.Cost) || math.IsInf(res.Cost, 0) {
		t.Fatalf("Cost is invalid with negative zero: %f", res.Cost)
	}
}

// ⚡ 4. Chaotic Non-Deterministic Sampling / Stochastic Noise Attack
func TestPenetration_4_ChaoticStochasticNoise(t *testing.T) {
	// Sampler returning heavy-tailed Cauchy noise
	cauchySampler := func(bSize int) [][]float64 {
		res := make([][]float64, bSize)
		for i := range res {
			// Standard Cauchy draw = tan(pi * (u - 0.5))
			u := rand.Float64()
			for u == 0.5 {
				u = rand.Float64()
			}
			val := math.Tan(math.Pi * (u - 0.5))
			res[i] = []float64{val}
		}
		return res
	}

	cfg := StreamingConfig{
		Epsilon:      0.5,
		BatchSize:    32,
		Steps:        20,
		NumFeatures:  32,
		LearningRate: 0.01,
	}

	cost, err := StreamingContinuousSinkhorn(cauchySampler, cauchySampler, 1, cfg)
	if err != nil {
		t.Fatalf("StreamingContinuousSinkhorn failed under chaotic noise: %v", err)
	}
	if math.IsNaN(cost) {
		t.Fatalf("Returned NaN cost under chaotic noise")
	}
}

// 🔄 5. Long-Running Drift & Memory Leak Profiling
func TestPenetration_5_MemoryDriftAndLeak(t *testing.T) {
	runtime.GC()
	var mStart runtime.MemStats
	runtime.ReadMemStats(&mStart)

	costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) }
	logR := []float64{-1.38, -1.38, -1.38, -1.38}
	logC := []float64{-1.38, -1.38, -1.38, -1.38}
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 50,
		Tolerance:     1e-4,
		EnableHistory: true,
	}

	for i := 0; i < 500; i++ {
		_, err := LogSinkhornStreaming(4, 4, logR, logC, costFn, cfg)
		if err != nil {
			t.Fatalf("Error in iteration %d: %v", i, err)
		}
	}

	runtime.GC()
	time.Sleep(10 * time.Millisecond)

	var mEnd runtime.MemStats
	runtime.ReadMemStats(&mEnd)

	// Check if HeapAlloc did not grow unnaturally (> 10 MB increase)
	allocDiff := int64(mEnd.HeapAlloc) - int64(mStart.HeapAlloc)
	if allocDiff > 10*1024*1024 {
		t.Errorf("Potential memory leak detected! HeapAlloc increased by %d bytes", allocDiff)
	}
}

// 🎯 6. Extreme Matrix Conditioning Number Attack
func TestPenetration_6_IllConditionedMatrix(t *testing.T) {
	// Construct Hilbert matrix H_ij = 1 / (i + j + 1) which is ill-conditioned
	n := 12
	H := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			H.Set(i, j, 1.0/float64(i+j+1))
		}
	}

	_, err := InvertSmallMatrix(H)
	if err == nil {
		t.Logf("Matrix inverted successfully without numerical divergence")
	} else {
		// Expecting dynamic pivot thresholding error for extreme ill-conditioning
		t.Logf("Ill-conditioned matrix properly rejected: %v", err)
	}
}

// 🧩 7. Non-Convex & Asymmetric Distance Metric Injections
func TestPenetration_7_AsymmetricNonConvexMetrics(t *testing.T) {
	ptsX := [][]float64{{0.0}, {1.0}}
	ptsY := [][]float64{{2.0}, {3.0}}

	// Asymmetric metric d(a, b) != d(b, a) violating triangle inequality
	asymCost := func(a, b []float64) float64 {
		diff := a[0] - b[0]
		if diff < 0 {
			return -diff * 3.0
		}
		return diff * 0.1
	}

	cfg := LogSinkhornConfig{
		Epsilon:       0.2,
		MaxIterations: 100,
		Tolerance:     1e-4,
	}

	div, err := SinkhornDivergence(ptsX, ptsY, asymCost, cfg)
	if err != nil {
		t.Fatalf("SinkhornDivergence failed on asymmetric metric: %v", err)
	}
	if div < 0 {
		t.Errorf("Debiased divergence returned negative value: %f", div)
	}
}

// 💣 8. Extreme Large Dimension & Memory Allocation Limit
func TestPenetration_8_ExtremelyLargeDimensionOverflow(t *testing.T) {
	M, N := 1000, 1000
	logR := make([]float64, M)
	logC := make([]float64, N)
	for i := range logR { logR[i] = -math.Log(float64(M)) }
	for j := range logC { logC[j] = -math.Log(float64(N)) }

	costFn := func(i, j int) float64 {
		return float64((i - j) * (i - j)) * 0.0001
	}

	cfg := LogSinkhornConfig{
		Epsilon:       0.2,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 5,
		Tolerance:     1e-3,
	}

	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming failed on large dimension: %v", err)
	}
	if len(res.F) != M || len(res.G) != N {
		t.Fatalf("Dimension mismatch for large matrix solve")
	}
}

// ☣️ 9. Infinity and NaN Injection in Cost Function
func TestPenetration_9_InfAndNaNInCostFunction(t *testing.T) {
	M, N := 4, 4
	logR := make([]float64, M)
	logC := make([]float64, N)
	for i := range logR { logR[i] = -math.Log(float64(M)) }
	for j := range logC { logC[j] = -math.Log(float64(N)) }

	// Cost function injecting +Inf and NaN
	costFn := func(i, j int) float64 {
		if i == 0 && j == 0 {
			return math.Inf(1)
		}
		if i == 1 && j == 1 {
			return math.NaN()
		}
		return float64((i - j) * (i - j))
	}

	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 10,
		Tolerance:     1e-4,
	}

	// Should execute without panicking
	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfg)
	if err != nil {
		t.Logf("Handled cost function Inf/NaN gracefully with error: %v", err)
	} else if res != nil {
		t.Logf("Handled cost function Inf/NaN with result cost: %f", res.Cost)
	}
}

// ❄️ 10. Extreme Epsilon Underflow/Overflow ($10^{-300}$)
func TestPenetration_10_EpsilonUnderflowInfinity(t *testing.T) {
	M, N := 3, 3
	logR := make([]float64, M)
	logC := make([]float64, N)
	for i := range logR { logR[i] = -math.Log(float64(M)) }
	for j := range logC { logC[j] = -math.Log(float64(N)) }

	costFn := func(i, j int) float64 { return float64(i + j) }

	// Extreme small epsilon close to float64 minimum
	cfg := LogSinkhornConfig{
		Epsilon:       1e-300,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 10,
		Tolerance:     1e-4,
	}

	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfg)
	if err != nil {
		t.Logf("Extreme epsilon handled with error: %v", err)
	} else if res != nil {
		t.Logf("Extreme epsilon execution completed, converged: %t", res.Converged)
	}
}

// ⚡ 11. Concurrent Context Cancellation Stress
func TestPenetration_11_ConcurrentContextCancellationRace(t *testing.T) {
	const goroutines = 20
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(rand.Intn(5))*time.Millisecond)
			defer cancel()

			M, N := 50, 50
			logR := make([]float64, M)
			logC := make([]float64, N)
			for i := range logR { logR[i] = -math.Log(float64(M)) }
			for j := range logC { logC[j] = -math.Log(float64(N)) }

			costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) }
			cfg := LogSinkhornConfig{
				Epsilon:       0.1,
				Tau1:          1.0,
				Tau2:          1.0,
				MaxIterations: 200,
				Tolerance:     1e-6,
			}

			_, _ = LogSinkhornStreamingContext(ctx, M, N, logR, logC, costFn, cfg, nil)
		}()
	}

	wg.Wait()
}
