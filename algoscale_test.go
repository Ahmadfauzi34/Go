package algoscale

import (
	"math"
	"testing"
)

func TestEpsilonAnnealing(t *testing.T) {
	M, N := 5, 5
	logR := make([]float64, M)
	logC := make([]float64, N)
	for i := range logR {
		logR[i] = -1.609
	}
	for j := range logC {
		logC[j] = -1.609
	}
	costFn := func(i, j int) float64 {
		return float64((i - j) * (i - j))
	}

	cfg := LogSinkhornConfig{
		Epsilon:       0.01,
		EpsilonInit:   0.5,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 100,
		Tolerance:     1e-4,
	}

	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming failed with annealing: %v", err)
	}
	if res == nil {
		t.Fatalf("Expected non-nil result")
	}
}

func TestJSONExportImport_ErrorPaths(t *testing.T) {
	var nilRes *LogSinkhornResult
	if _, err := nilRes.ExportJSON(); err == nil {
		t.Errorf("Expected error for nil ExportJSON receiver")
	}

	if _, err := ImportJSON([]byte("{invalid json")); err == nil {
		t.Errorf("Expected error for malformed JSON in ImportJSON")
	}
}

func TestSinkhornDivergence_ErrorPaths(t *testing.T) {
	euclid := func(a, b []float64) float64 { return (a[0]-b[0])*(a[0]-b[0]) }
	cfg := LogSinkhornConfig{Epsilon: 0.1, MaxIterations: 10, Tolerance: 1e-3, Tau1: 1, Tau2: 1}

	// Empty point cloud
	if _, err := SinkhornDivergence([][]float64{}, [][]float64{{1.0}}, euclid, cfg); err == nil {
		t.Errorf("Expected error for empty point cloud X")
	}

	// Error propagation from Sinkhorn solver (invalid epsilon <= 0)
	cfgInvalid := cfg
	cfgInvalid.Epsilon = -0.1
	if _, err := SinkhornDivergence([][]float64{{1.0}}, [][]float64{{2.0}}, euclid, cfgInvalid); err == nil {
		t.Errorf("Expected error propagation for negative Epsilon")
	}
}

func TestComputeBarycenterLogDomain_UncoveredBranches(t *testing.T) {
	costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) }
	baryCfg := BarycenterConfig{Epsilon: 0.5, MaxIterations: 10, Tolerance: 1e-3}

	// Empty distributions K = 0
	if _, err := ComputeBarycenterLogDomain(nil, nil, costFn, 2, 2, baryCfg); err == nil {
		t.Errorf("Expected error for empty distributions")
	}

	// Weight length mismatch
	dists := [][]float64{{0.5, 0.5}, {0.2, 0.8}}
	if _, err := ComputeBarycenterLogDomain(dists, []float64{1.0}, costFn, 2, 2, baryCfg); err == nil {
		t.Errorf("Expected error for weight length mismatch")
	}

	// M <= 0 or N <= 0
	if _, err := ComputeBarycenterLogDomain(dists, []float64{0.5, 0.5}, costFn, 0, 2, baryCfg); err == nil {
		t.Errorf("Expected error for M <= 0")
	}

	// Invalid config
	badCfg := baryCfg
	badCfg.Epsilon = 0
	if _, err := ComputeBarycenterLogDomain(dists, []float64{0.5, 0.5}, costFn, 2, 2, badCfg); err == nil {
		t.Errorf("Expected error for Epsilon <= 0")
	}

	// Weight sum != 1
	if _, err := ComputeBarycenterLogDomain(dists, []float64{0.3, 0.3}, costFn, 2, 2, baryCfg); err == nil {
		t.Errorf("Expected error for sum(weights) != 1.0")
	}

	// Invalid weights (negative or NaN)
	if _, err := ComputeBarycenterLogDomain(dists, []float64{math.NaN(), 1.0}, costFn, 2, 2, baryCfg); err == nil {
		t.Errorf("Expected error for NaN weight")
	}

	// Distribution length != N
	badDists := [][]float64{{0.5}, {0.2, 0.8}}
	if _, err := ComputeBarycenterLogDomain(badDists, []float64{0.5, 0.5}, costFn, 2, 2, baryCfg); err == nil {
		t.Errorf("Expected error for distribution length != N")
	}

	// Negative / NaN distribution entries
	negDists := [][]float64{{-0.5, 0.5}, {0.2, 0.8}}
	if _, err := ComputeBarycenterLogDomain(negDists, []float64{0.5, 0.5}, costFn, 2, 2, baryCfg); err == nil {
		t.Errorf("Expected error for negative distribution entry")
	}

	// Zero sum distribution
	zeroDists := [][]float64{{0.0, 0.0}, {0.2, 0.8}}
	if _, err := ComputeBarycenterLogDomain(zeroDists, []float64{0.5, 0.5}, costFn, 2, 2, baryCfg); err == nil {
		t.Errorf("Expected error for zero sum distribution")
	}

	// Zero weight branch (weights[k] == 0.0)
	dists3 := [][]float64{{0.5, 0.5}, {0.2, 0.8}, {0.1, 0.9}}
	weights3 := []float64{0.5, 0.5, 0.0}
	resZeroWeight, err := ComputeBarycenterLogDomain(dists3, weights3, costFn, 2, 2, baryCfg)
	if err != nil {
		t.Fatalf("ComputeBarycenterLogDomain with zero weight failed: %v", err)
	}
	if resZeroWeight == nil {
		t.Fatalf("Expected non-nil result for zero weight branch")
	}

	// Single distribution branch K = 1
	dists1 := [][]float64{{0.4, 0.6}}
	weights1 := []float64{1.0}
	resK1, err := ComputeBarycenterLogDomain(dists1, weights1, costFn, 2, 2, baryCfg)
	if err != nil {
		t.Fatalf("ComputeBarycenterLogDomain with K=1 failed: %v", err)
	}
	if resK1 == nil {
		t.Fatalf("Expected non-nil result for K=1")
	}
}

func TestStreamingContinuousSinkhorn_DefaultNumFeatures(t *testing.T) {
	sX := func(n int) [][]float64 {
		res := make([][]float64, n)
		for i := range res { res[i] = []float64{0.1} }
		return res
	}

	// Default NumFeatures (NumFeatures <= 0)
	cfg := StreamingConfig{
		Epsilon:      0.2,
		BatchSize:    10,
		Steps:        2,
		NumFeatures:  0, // Should default to 256
		LearningRate: 0.01,
	}

	cost, err := StreamingContinuousSinkhorn(sX, sX, 1, cfg)
	if err != nil {
		t.Fatalf("StreamingContinuousSinkhorn failed with default NumFeatures: %v", err)
	}
	if math.IsNaN(cost) {
		t.Errorf("Expected non-NaN cost")
	}
}

func TestLogSumExpWeighted(t *testing.T) {
	// Normal case
	vals := []float64{1.0, 2.0, 3.0}
	logW := []float64{0.0, 0.0, 0.0} // log(1) = 0, weights = 1,1,1
	res := LogSumExpWeighted(vals, logW)
	expected := LogSumExp(vals)
	if math.Abs(res-expected) > 1e-9 {
		t.Errorf("LogSumExpWeighted mismatched with LogSumExp: got %f, want %f", res, expected)
	}

	// Empty slice
	if !math.IsInf(LogSumExpWeighted(nil, nil), -1) {
		t.Errorf("Expected -Inf for empty slices in LogSumExpWeighted")
	}

	// Dimension mismatch
	if !math.IsInf(LogSumExpWeighted([]float64{1.0}, []float64{1.0, 2.0}), -1) {
		t.Errorf("Expected -Inf for mismatched slice lengths")
	}

	// All -Inf
	if !math.IsInf(LogSumExpWeighted([]float64{math.Inf(-1)}, []float64{0.0}), -1) {
		t.Errorf("Expected -Inf when all terms are -Inf")
	}
}

func TestLowRankKernel_NewAndSolve(t *testing.T) {
	M, N, Rank := 4, 4, 2
	Kxz := NewMatrix(M, Rank)
	Kzy := NewMatrix(Rank, N)
	Kzz := NewMatrix(Rank, Rank)

	// Fill identity-like matrices
	Kxz.Set(0, 0, 1.0); Kxz.Set(1, 0, 0.5); Kxz.Set(2, 1, 0.5); Kxz.Set(3, 1, 1.0)
	Kzy.Set(0, 0, 1.0); Kzy.Set(0, 1, 0.5); Kzy.Set(1, 2, 0.5); Kzy.Set(1, 3, 1.0)
	Kzz.Set(0, 0, 2.0); Kzz.Set(0, 1, 0.0); Kzz.Set(1, 0, 0.0); Kzz.Set(1, 1, 2.0)

	lrk, err := NewLowRankKernel(M, N, Rank, Kxz, Kzz, Kzy)
	if err != nil {
		t.Fatalf("NewLowRankKernel failed: %v", err)
	}

	r := []float64{0.25, 0.25, 0.25, 0.25}
	c := []float64{0.25, 0.25, 0.25, 0.25}

	res, err := lrk.Solve(r, c, 100, 1e-5)
	if err != nil {
		t.Fatalf("LowRankKernel.Solve failed: %v", err)
	}

	if res == nil || len(res.U) != M || len(res.V) != N {
		t.Fatalf("Invalid LowRankResult: %+v", res)
	}

	// Error paths
	// Nil receiver
	var nilLRK *LowRankKernel
	if _, err := nilLRK.Solve(r, c, 10, 1e-3); err == nil {
		t.Errorf("Expected error for nil LowRankKernel receiver")
	}

	// Nil inputs to NewLowRankKernel
	if _, err := NewLowRankKernel(M, N, Rank, nil, Kzz, Kzy); err == nil {
		t.Errorf("Expected error for nil matrix in NewLowRankKernel")
	}

	// Invalid dimensions
	if _, err := NewLowRankKernel(0, N, Rank, Kxz, Kzz, Kzy); err == nil {
		t.Errorf("Expected error for non-positive dimension M")
	}

	badKxz := NewMatrix(M+1, Rank)
	if _, err := NewLowRankKernel(M, N, Rank, badKxz, Kzz, Kzy); err == nil {
		t.Errorf("Expected error for Kxz dimension mismatch")
	}

	badKzy := NewMatrix(Rank, N+1)
	if _, err := NewLowRankKernel(M, N, Rank, Kxz, Kzz, badKzy); err == nil {
		t.Errorf("Expected error for Kzy dimension mismatch")
	}

	badKzz := NewMatrix(Rank, Rank+1)
	if _, err := NewLowRankKernel(M, N, Rank, Kxz, badKzz, Kzy); err == nil {
		t.Errorf("Expected error for Kzz non-square")
	}

	// Singular Kzz
	zeroKzz := NewMatrix(Rank, Rank)
	if _, err := NewLowRankKernel(M, N, Rank, Kxz, zeroKzz, Kzy); err == nil {
		t.Errorf("Expected error for singular Kzz matrix")
	}

	// Solve dimension / parameter mismatch
	if _, err := lrk.Solve([]float64{0.5}, c, 100, 1e-5); err == nil {
		t.Errorf("Expected error for marginal r dimension mismatch")
	}

	if _, err := lrk.Solve(r, c, 0, 1e-5); err == nil {
		t.Errorf("Expected error for maxIter <= 0")
	}

	if _, err := lrk.Solve([]float64{-1.0, 0.25, 0.25, 0.25}, c, 100, 1e-5); err == nil {
		t.Errorf("Expected error for negative marginal r entry")
	}
}

func TestLogSinkhornParallel(t *testing.T) {
	M, N := 120, 120
	logR := make([]float64, M)
	logC := make([]float64, N)
	for i := range logR { logR[i] = -math.Log(float64(M)) }
	for j := range logC { logC[j] = -math.Log(float64(N)) }

	costFn := func(i, j int) float64 {
		diff := float64(i - j)
		return diff * diff * 0.01
	}

	cfg := LogSinkhornConfig{
		Epsilon:         0.1,
		Tau1:            1.0,
		Tau2:            1.0,
		MaxIterations:   50,
		Tolerance:       1e-5,
		SparseThreshold: 5.0,
	}

	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming parallel execution failed: %v", err)
	}
	if !res.Converged {
		t.Errorf("Expected parallel LogSinkhorn to converge")
	}
	if len(res.F) != M || len(res.G) != N {
		t.Errorf("Result dimension mismatch: len(F)=%d, len(G)=%d", len(res.F), len(res.G))
	}
}

func TestStreamingContinuousSinkhorn_EdgeCases(t *testing.T) {
	// Nil sampler check
	cfg := StreamingConfig{
		Epsilon:      0.1,
		BatchSize:    10,
		Steps:        5,
		NumFeatures:  16,
		LearningRate: 0.01,
	}
	if _, err := StreamingContinuousSinkhorn(nil, nil, 2, cfg); err == nil {
		t.Errorf("Expected error when samplers are nil")
	}

	// Invalid dimensions or config parameters
	dummySampler := func(b int) [][]float64 { return make([][]float64, b) }
	if _, err := StreamingContinuousSinkhorn(dummySampler, dummySampler, 0, cfg); err == nil {
		t.Errorf("Expected error for dim <= 0")
	}

	cfgInvalid := cfg
	cfgInvalid.BatchSize = 0
	if _, err := StreamingContinuousSinkhorn(dummySampler, dummySampler, 2, cfgInvalid); err == nil {
		t.Errorf("Expected error for BatchSize <= 0")
	}
}

func TestMatrixAndInvert_EdgeCases(t *testing.T) {
	// NewMatrix panic on non-positive dimensions
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for NewMatrix(0, 0)")
		}
	}()
	_ = NewMatrix(0, 0)
}

func TestInvertSmallMatrix_Validation(t *testing.T) {
	// Nil matrix
	if _, err := InvertSmallMatrix(nil); err == nil {
		t.Errorf("Expected error for nil Matrix input in InvertSmallMatrix")
	}

	// Non-square matrix
	nonSquare := &Matrix{Rows: 2, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6}}
	if _, err := InvertSmallMatrix(nonSquare); err == nil {
		t.Errorf("Expected error for non-square Matrix input")
	}

	// Zero-norm (all zeros) matrix
	zeroMat := NewMatrix(2, 2)
	if _, err := InvertSmallMatrix(zeroMat); err == nil {
		t.Errorf("Expected error for all-zero Matrix input")
	}
}

func TestUnbalancedSinkhorn_Validation(t *testing.T) {
	K := NewMatrix(2, 2)

	// Nil kernel
	if _, err := UnbalancedSinkhorn(nil, []float64{0.5, 0.5}, []float64{0.5, 0.5}, UnbalancedConfig{Epsilon: 0.1, Tau1: 1, Tau2: 1, MaxIterations: 10, Tolerance: 1e-3}); err == nil {
		t.Errorf("Expected error for nil kernel K")
	}

	// Marginal dimension mismatch
	if _, err := UnbalancedSinkhorn(K, []float64{0.5}, []float64{0.5, 0.5}, UnbalancedConfig{Epsilon: 0.1, Tau1: 1, Tau2: 1, MaxIterations: 10, Tolerance: 1e-3}); err == nil {
		t.Errorf("Expected error for marginal r dimension mismatch")
	}

	// Non-positive hyperparameters
	if _, err := UnbalancedSinkhorn(K, []float64{0.5, 0.5}, []float64{0.5, 0.5}, UnbalancedConfig{Epsilon: -0.1, Tau1: 1, Tau2: 1, MaxIterations: 10, Tolerance: 1e-3}); err == nil {
		t.Errorf("Expected error for Epsilon <= 0")
	}

	// Invalid marginal values (NaN or negative)
	if _, err := UnbalancedSinkhorn(K, []float64{math.NaN(), 0.5}, []float64{0.5, 0.5}, UnbalancedConfig{Epsilon: 0.1, Tau1: 1, Tau2: 1, MaxIterations: 10, Tolerance: 1e-3}); err == nil {
		t.Errorf("Expected error for NaN in marginal r")
	}
}

func TestInvertSmallMatrix_ZeroDimension(t *testing.T) {
	mat := &Matrix{Rows: 0, Cols: 0, Data: []float64{}}
	if _, err := InvertSmallMatrix(mat); err == nil {
		t.Errorf("Expected error for zero dimension matrix")
	}
}

func TestLogSumExp_UncoveredBranches(t *testing.T) {
	// Empty slice
	if !math.IsInf(LogSumExp(nil), -1) {
		t.Errorf("Expected -Inf for empty slice")
	}

	// Slices where maxVal is -Inf
	negInfs := []float64{math.Inf(-1), math.Inf(-1)}
	if !math.IsInf(LogSumExp(negInfs), -1) {
		t.Errorf("Expected -Inf when all elements are -Inf")
	}
}

func TestUnbalancedSinkhorn_UncoveredBranches(t *testing.T) {
	K := NewMatrix(2, 2)
	// Column marginal mismatch len(c) != n
	if _, err := UnbalancedSinkhorn(K, []float64{0.5, 0.5}, []float64{0.5}, UnbalancedConfig{Epsilon: 0.1, Tau1: 1, Tau2: 1, MaxIterations: 10, Tolerance: 1e-3}); err == nil {
		t.Errorf("Expected error for column marginal length mismatch")
	}

	// Invalid MaxIterations and Tolerance <= 0
	if _, err := UnbalancedSinkhorn(K, []float64{0.5, 0.5}, []float64{0.5, 0.5}, UnbalancedConfig{Epsilon: 0.1, Tau1: 1, Tau2: 1, MaxIterations: 0, Tolerance: 1e-3}); err == nil {
		t.Errorf("Expected error for MaxIterations <= 0")
	}

	// Invalid c marginal entries (negative / NaN)
	if _, err := UnbalancedSinkhorn(K, []float64{0.5, 0.5}, []float64{-1.0, 0.5}, UnbalancedConfig{Epsilon: 0.1, Tau1: 1, Tau2: 1, MaxIterations: 10, Tolerance: 1e-3}); err == nil {
		t.Errorf("Expected error for negative c marginal entry")
	}

	// Zero-kernel matrix (tempKv <= 1e-300 and tempKtu <= 1e-300)
	zeroK := NewMatrix(2, 2)
	resZero, err := UnbalancedSinkhorn(zeroK, []float64{0.5, 0.5}, []float64{0.5, 0.5}, UnbalancedConfig{Epsilon: 0.1, Tau1: 1, Tau2: 1, MaxIterations: 10, Tolerance: 1e-3})
	if err != nil {
		t.Fatalf("UnbalancedSinkhorn with zero kernel failed: %v", err)
	}
	if resZero.U[0] != 0.0 || resZero.V[0] != 0.0 {
		t.Errorf("Expected zero vectors for zero kernel updates")
	}
}

func TestLowRankKernel_SolveUncoveredBranches(t *testing.T) {
	M, N, Rank := 2, 2, 1
	Kxz := NewMatrix(M, Rank)
	Kzy := NewMatrix(Rank, N)
	Kzz := NewMatrix(Rank, Rank)

	Kxz.Set(0, 0, 1.0); Kxz.Set(1, 0, 1.0)
	Kzy.Set(0, 0, 1.0); Kzy.Set(0, 1, 1.0)
	Kzz.Set(0, 0, 1.0)

	lrk, err := NewLowRankKernel(M, N, Rank, Kxz, Kzz, Kzy)
	if err != nil {
		t.Fatalf("NewLowRankKernel failed: %v", err)
	}

	r := []float64{0.5, 0.5}
	c := []float64{0.5, 0.5}

	// Column marginal mismatch len(c) != n
	if _, err := lrk.Solve(r, []float64{0.5}, 10, 1e-3); err == nil {
		t.Errorf("Expected error for len(c) mismatch")
	}

	// Invalid marginal c (negative / NaN)
	if _, err := lrk.Solve(r, []float64{0.5, -0.5}, 10, 1e-3); err == nil {
		t.Errorf("Expected error for negative c entry")
	}

	// Zero kernel matrices for kv <= 1e-300 and ktu <= 1e-300
	KxzZero := NewMatrix(M, Rank)
	lrkZero, _ := NewLowRankKernel(M, N, Rank, KxzZero, Kzz, Kzy)
	resZero, err := lrkZero.Solve(r, c, 10, 1e-3)
	if err != nil {
		t.Fatalf("LowRankKernel.Solve with zero kernel failed: %v", err)
	}
	if resZero.U[0] != 0.0 || resZero.V[0] != 0.0 {
		t.Errorf("Expected u=0 and v=0 for zero kernel")
	}
}

func TestLogSinkhornStreamingContext_UncoveredBranches(t *testing.T) {
	costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) }

	// Invalid M or N
	if _, err := LogSinkhornStreaming(0, 2, []float64{}, []float64{0, 0}, costFn, LogSinkhornConfig{Epsilon: 0.1, MaxIterations: 10, Tolerance: 1e-3, Tau1: 1, Tau2: 1}); err == nil {
		t.Errorf("Expected error for M <= 0")
	}

	// Marginal length mismatch
	if _, err := LogSinkhornStreaming(2, 2, []float64{0}, []float64{0, 0}, costFn, LogSinkhornConfig{Epsilon: 0.1, MaxIterations: 10, Tolerance: 1e-3, Tau1: 1, Tau2: 1}); err == nil {
		t.Errorf("Expected error for logR length mismatch")
	}

	// Invalid Epsilon / MaxIterations / Tolerance
	logR2 := []float64{-0.693, -0.693}
	logC2 := []float64{-0.693, -0.693}
	if _, err := LogSinkhornStreaming(2, 2, logR2, logC2, costFn, LogSinkhornConfig{Epsilon: -0.1, MaxIterations: 10, Tolerance: 1e-3, Tau1: 1, Tau2: 1}); err == nil {
		t.Errorf("Expected error for Epsilon <= 0")
	}

	// Invalid Tau1 / Tau2
	if _, err := LogSinkhornStreaming(2, 2, logR2, logC2, costFn, LogSinkhornConfig{Epsilon: 0.1, MaxIterations: 10, Tolerance: 1e-3, Tau1: 0, Tau2: 1}); err == nil {
		t.Errorf("Expected error for Tau1 <= 0")
	}

	// Infinite Tau1 and Tau2 with Epsilon Annealing
	cfgAnnealInf := LogSinkhornConfig{
		Epsilon:       0.01,
		EpsilonInit:   0.5,
		Tau1:          math.Inf(1),
		Tau2:          math.Inf(1),
		MaxIterations: 10,
		Tolerance:     1e-3,
	}
	resInf, err := LogSinkhornStreaming(2, 2, logR2, logC2, costFn, cfgAnnealInf)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming with infinite tau and annealing failed: %v", err)
	}
	if resInf == nil {
		t.Fatalf("Expected non-nil result")
	}
}
