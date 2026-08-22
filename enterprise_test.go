package algoscale

import (
	"context"
	"math"
	"testing"
	"time"
)

type mockObserver struct {
	called        bool
	iterations    int
	finalResidual float64
	converged     bool
}

func (m *mockObserver) ObserveSolve(iterations int, duration time.Duration, finalResidual float64, converged bool) {
	m.called = true
	m.iterations = iterations
	m.finalResidual = finalResidual
	m.converged = converged
}

func TestEnterprise_JSONExportImport(t *testing.T) {
	res := &LogSinkhornResult{
		F:               []float64{1.0, 2.0},
		G:               []float64{3.0, 4.0},
		Iterations:      15,
		FinalResidual:   1e-5,
		ResidualHistory: []float64{0.1, 0.01, 1e-5},
		Converged:       true,
		Cost:            12.34,
	}

	data, err := res.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON failed: %v", err)
	}

	imported, err := ImportJSON(data)
	if err != nil {
		t.Fatalf("ImportJSON failed: %v", err)
	}

	if imported.Iterations != res.Iterations || imported.Cost != res.Cost || !imported.Converged {
		t.Errorf("Imported JSON mismatch: %+v vs %+v", imported, res)
	}
}

func TestEnterprise_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	M, N := 4, 4
	logR := make([]float64, M)
	logC := make([]float64, N)
	costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) }
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 100,
		Tolerance:     1e-6,
	}

	_, err := LogSinkhornStreamingContext(ctx, M, N, logR, logC, costFn, cfg, nil)
	if err == nil {
		t.Fatalf("Expected context cancellation error, got nil")
	}
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
}

func TestEnterprise_MetricsObserver(t *testing.T) {
	M, N := 2, 2
	logR := []float64{-math.Log(2), -math.Log(2)}
	logC := []float64{-math.Log(2), -math.Log(2)}
	costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) }
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 20,
		Tolerance:     1e-4,
	}

	obs := &mockObserver{}
	res, err := LogSinkhornStreamingContext(context.Background(), M, N, logR, logC, costFn, cfg, obs)
	if err != nil {
		t.Fatalf("LogSinkhornStreamingContext failed: %v", err)
	}

	if !obs.called {
		t.Errorf("Expected observer to be called")
	}
	if obs.iterations != res.Iterations {
		t.Errorf("Observer iterations mismatch: got %d, expected %d", obs.iterations, res.Iterations)
	}
}
