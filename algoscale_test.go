package algoscale

import (
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
