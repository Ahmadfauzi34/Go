package algoscale

import (
	"testing"
)

func TestObservability(t *testing.T) {
	M, N := 4, 4
	logR := make([]float64, M)
	logC := make([]float64, N)
	for i := range logR {
		logR[i] = -1.386
	}
	for j := range logC {
		logC[j] = -1.386
	}
	costFn := func(i, j int) float64 {
		return float64((i - j) * (i - j))
	}

	var callbackCalls int
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 10,
		Tolerance:     1e-8,
		EnableHistory: true,
		OnIteration: func(iter int, residual float64, eps float64) {
			callbackCalls++
		},
	}

	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming failed: %v", err)
	}

	if len(res.ResidualHistory) == 0 {
		t.Errorf("Expected non-empty residual history")
	}
	if callbackCalls == 0 {
		t.Errorf("Expected callback calls to be > 0")
	}
}
