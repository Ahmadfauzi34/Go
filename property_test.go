package algoscale

import (
	"math"
	"testing"
)

// Property 1: Distance Identity S_eps(X, X) ≈ 0
func TestProperty_DistanceIdentity(t *testing.T) {
	pts := [][]float64{
		{0.0, 0.0},
		{1.0, 1.0},
		{0.5, 0.5},
	}
	euclid := func(a, b []float64) float64 {
		return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])
	}
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		MaxIterations: 200,
		Tolerance:     1e-6,
	}

	div, err := SinkhornDivergence(pts, pts, euclid, cfg)
	if err != nil {
		t.Fatalf("SinkhornDivergence error: %v", err)
	}
	if math.Abs(div) > 1e-4 {
		t.Errorf("Expected S_eps(X, X) approx 0, got %f", div)
	}
}

// Property 2: Symmetry S_eps(X, Y) = S_eps(Y, X)
func TestProperty_Symmetry(t *testing.T) {
	ptsX := [][]float64{{0.0, 0.0}, {1.0, 0.0}}
	ptsY := [][]float64{{0.0, 1.0}, {1.0, 1.0}}
	euclid := func(a, b []float64) float64 {
		return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])
	}
	cfg := LogSinkhornConfig{
		Epsilon:       0.2,
		MaxIterations: 200,
		Tolerance:     1e-6,
	}

	divXY, errXY := SinkhornDivergence(ptsX, ptsY, euclid, cfg)
	if errXY != nil {
		t.Fatalf("SinkhornDivergence XY error: %v", errXY)
	}
	divYX, errYX := SinkhornDivergence(ptsY, ptsX, euclid, cfg)
	if errYX != nil {
		t.Fatalf("SinkhornDivergence YX error: %v", errYX)
	}

	if math.Abs(divXY-divYX) > 1e-5 {
		t.Errorf("Symmetry violated: S_eps(X, Y) = %f, S_eps(Y, X) = %f", divXY, divYX)
	}
}

// Property 3: Non-negativity Debiased Divergence S_eps(X, Y) >= 0
func TestProperty_NonNegativity(t *testing.T) {
	ptsX := [][]float64{{0.1, 0.2}, {0.5, 0.8}}
	ptsY := [][]float64{{0.9, 0.4}, {0.3, 0.1}}
	euclid := func(a, b []float64) float64 {
		return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])
	}
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		MaxIterations: 200,
		Tolerance:     1e-6,
	}

	div, err := SinkhornDivergence(ptsX, ptsY, euclid, cfg)
	if err != nil {
		t.Fatalf("SinkhornDivergence error: %v", err)
	}
	if div < 0 {
		t.Errorf("Expected S_eps(X, Y) >= 0, got %f", div)
	}
}

// Property 4: Transport Plan Mass Conservation
func TestProperty_MassConservation(t *testing.T) {
	m, n := 3, 4
	K := NewMatrix(m, n)
	for i := 0; i < m*n; i++ {
		K.Data[i] = 0.5
	}
	r := []float64{0.2, 0.5, 0.3}
	c := []float64{0.1, 0.4, 0.3, 0.2}

	cfg := UnbalancedConfig{
		Epsilon:       0.5,
		Tau1:          1e5, // close to balanced
		Tau2:          1e5,
		MaxIterations: 300,
		Tolerance:     1e-6,
	}

	res, err := UnbalancedSinkhorn(K, r, c, cfg)
	if err != nil {
		t.Fatalf("UnbalancedSinkhorn error: %v", err)
	}

	for i := 0; i < m; i++ {
		rowSum := 0.0
		for j := 0; j < n; j++ {
			rowSum += res.Plan.At(i, j)
		}
		if math.Abs(rowSum-r[i]) > 1e-2 {
			t.Errorf("Row mass mismatch at row %d: got %f, want %f", i, rowSum, r[i])
		}
	}

	for j := 0; j < n; j++ {
		colSum := 0.0
		for i := 0; i < m; i++ {
			colSum += res.Plan.At(i, j)
		}
		if math.Abs(colSum-c[j]) > 1e-2 {
			t.Errorf("Col mass mismatch at col %d: got %f, want %f", j, colSum, c[j])
		}
	}
}
