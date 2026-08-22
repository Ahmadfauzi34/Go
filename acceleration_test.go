package algoscale

import (
	"math"
	"testing"
)

func TestAcceleration_Momentum(t *testing.T) {
	M, N := 6, 6
	logR := make([]float64, M)
	logC := make([]float64, N)
	for i := range logR { logR[i] = -math.Log(float64(M)) }
	for j := range logC { logC[j] = -math.Log(float64(N)) }

	costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) * 0.1 }

	cfgStandard := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 100,
		Tolerance:     1e-6,
	}

	resStandard, err1 := LogSinkhornStreaming(M, N, logR, logC, costFn, cfgStandard)
	if err1 != nil {
		t.Fatalf("Standard Sinkhorn failed: %v", err1)
	}

	cfgMomentum := cfgStandard
	cfgMomentum.MomentumBeta = 0.2

	resMomentum, err2 := LogSinkhornStreaming(M, N, logR, logC, costFn, cfgMomentum)
	if err2 != nil {
		t.Fatalf("Momentum Sinkhorn failed: %v", err2)
	}

	if !resMomentum.Converged {
		t.Errorf("Expected momentum Sinkhorn to converge")
	}

	if math.Abs(resStandard.Cost-resMomentum.Cost) > 1e-3 {
		t.Errorf("Cost mismatch between standard (%f) and momentum (%f)", resStandard.Cost, resMomentum.Cost)
	}
}

func TestAcceleration_SparseTruncation(t *testing.T) {
	M, N := 5, 5
	logR := make([]float64, M)
	logC := make([]float64, N)
	for i := range logR { logR[i] = -math.Log(float64(M)) }
	for j := range logC { logC[j] = -math.Log(float64(N)) }

	// High costs for distant entries
	costFn := func(i, j int) float64 {
		diff := float64(i - j)
		if diff < 0 { diff = -diff }
		if diff > 2 { return 100.0 }
		return diff * 0.1
	}

	cfgSparse := LogSinkhornConfig{
		Epsilon:         0.1,
		Tau1:            1.0,
		Tau2:            1.0,
		MaxIterations:   50,
		Tolerance:       1e-4,
		SparseThreshold: 10.0,
	}

	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfgSparse)
	if err != nil {
		t.Fatalf("Sparse truncation Sinkhorn failed: %v", err)
	}
	if !res.Converged {
		t.Errorf("Expected sparse truncation Sinkhorn to converge")
	}
}
