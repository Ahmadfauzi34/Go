package algoscale

import (
	"math"
	"testing"
)

func TestUtilities_StagnationEarlyStopping(t *testing.T) {
	M, N := 4, 4
	logR := make([]float64, M)
	logC := make([]float64, N)
	for i := range logR { logR[i] = -math.Log(float64(M)) }
	for j := range logC { logC[j] = -math.Log(float64(N)) }

	costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) }

	cfgStagnation := LogSinkhornConfig{
		Epsilon:          0.1,
		Tau1:             1.0,
		Tau2:             1.0,
		MaxIterations:    200,
		Tolerance:        1e-15, // extremely tight
		EnableHistory:    true,
		StagnationWindow: 3,
		StagnationTol:    1e-3,
	}

	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfgStagnation)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming failed: %v", err)
	}

	if res.Iterations >= 200 {
		t.Errorf("Expected early stopping before max iterations, got %d iterations", res.Iterations)
	}
}

func TestUtilities_ToDenseMatrix(t *testing.T) {
	M, N := 2, 2
	logR := []float64{-math.Log(2.0), -math.Log(2.0)}
	logC := []float64{-math.Log(2.0), -math.Log(2.0)}
	costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) }

	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 100,
		Tolerance:     1e-6,
	}

	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming failed: %v", err)
	}

	P, errP := res.ToDenseMatrix(M, N, logR, logC, costFn, 0.1)
	if errP != nil {
		t.Fatalf("ToDenseMatrix failed: %v", errP)
	}

	if P.Rows != 2 || P.Cols != 2 {
		t.Errorf("Expected 2x2 matrix, got %dx%d", P.Rows, P.Cols)
	}

	totalSum := 0.0
	for r := 0; r < M; r++ {
		for c := 0; c < N; c++ {
			totalSum += P.At(r, c)
		}
	}
	if totalSum <= 0 || math.IsNaN(totalSum) {
		t.Errorf("Total transport plan sum invalid: %f", totalSum)
	}
}
