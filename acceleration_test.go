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

// ============================================================================
// PENDALAMAN: VARIAN ACCELERATION (MomentumBeta & SparseThreshold)
//
// Batasan domain (anti-overlap):
// - TIDAK menguji EpsilonInit/annealing (domain algoscale_test.go)
// - TIDAK menguji JSON/observer/context, fuzz, property, penetration, utilities
// Semua assertion berbasis invariant implementasi:
// hasMomentum = (0 < beta < 1) dan hasSparse = (threshold > 0).
// ============================================================================

// Helper berprefix "accel" untuk menghindari bentrok identifier
// dengan file test lain (algoscale_test.go memakai prefix "annealing").
func accelLogUniform(n int) []float64 {
	out := make([]float64, n)
	v := -math.Log(float64(n))
	for i := range out {
		out[i] = v
	}
	return out
}

func accelBandCost(i, j int) float64 {
	diff := float64(i - j)
	if diff < 0 { diff = -diff }
	if diff > 2 { return 100.0 }
	return diff * 0.1
}

func accelAssertFinite(t *testing.T, label string, vals []float64) {
	t.Helper()
	for idx, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s[%d] tidak finite: %v", label, idx, v)
		}
	}
}

func accelCapture(t *testing.T, M, N int, logR, logC []float64, costFn CostFunction, cfg LogSinkhornConfig) (*LogSinkhornResult, []float64) {
	t.Helper()
	var seq []float64
	cfg.OnIteration = func(_ int, residual float64, _ float64) {
		seq = append(seq, residual)
	}
	res, err := LogSinkhornStreaming(M, N, logR, logC, costFn, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming gagal: %v", err)
	}
	return res, seq
}

func accelSeqMaxDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	maxDiff := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}

// [A] Momentum harus NON-aktif di luar interval terbuka (0,1):
// trajectory residual identik bitwise dengan baseline beta=0.
func TestAcceleration_MomentumInactiveOutsideOpenInterval(t *testing.T) {
	M, N := 6, 6
	logR := accelLogUniform(M)
	logC := accelLogUniform(N)
	costFn := func(i, j int) float64 { return float64((i-j)*(i-j)) * 0.1 }
	base := LogSinkhornConfig{Epsilon: 0.1, Tau1: 1.0, Tau2: 1.0, MaxIterations: 100, Tolerance: 1e-6}
	_, seqBase := accelCapture(t, M, N, logR, logC, costFn, base)
	for _, beta := range []float64{-0.5, 1.0, 1.5} {
		cfg := base
		cfg.MomentumBeta = beta
		_, seq := accelCapture(t, M, N, logR, logC, costFn, cfg)
		if d := accelSeqMaxDiff(seqBase, seq); d != 0 {
			t.Fatalf("beta=%v: momentum harus non-aktif, trajectory berbeda (maxDiff=%e)", beta, d)
		}
	}
}

// [B] Momentum aktif harus mengubah trajectory, tetapi titik tetap
// (cost akhir) tetap sama karena term ekstrapolasi lenyap saat konvergen.
func TestAcceleration_MomentumAltersTrajectorySameFixedPoint(t *testing.T) {
	M, N := 6, 6
	logR := accelLogUniform(M)
	logC := accelLogUniform(N)
	costFn := func(i, j int) float64 { return float64((i-j)*(i-j)) * 0.1 }
	base := LogSinkhornConfig{Epsilon: 0.1, Tau1: 1.0, Tau2: 1.0, MaxIterations: 100, Tolerance: 1e-6}
	resBase, seqBase := accelCapture(t, M, N, logR, logC, costFn, base)
	cfgMom := base
	cfgMom.MomentumBeta = 0.2
	resMom, seqMom := accelCapture(t, M, N, logR, logC, costFn, cfgMom)
	if d := accelSeqMaxDiff(seqBase, seqMom); d < 1e-6 {
		t.Fatalf("momentum aktif harus mengubah trajectory (maxDiff=%e)", d)
	}
	if !resMom.Converged {
		t.Fatalf("momentum beta=0.2 tidak konvergen")
	}
	if math.Abs(resBase.Cost-resMom.Cost) > 1e-3 {
		t.Fatalf("cost fixed-point berbeda: standard=%f momentum=%f", resBase.Cost, resMom.Cost)
	}
}

// [C] Sweep beta moderat: semua harus konvergen ke cost yang sama.
func TestAcceleration_MomentumBetaSweepConvergence(t *testing.T) {
	M, N := 6, 6
	logR := accelLogUniform(M)
	logC := accelLogUniform(N)
	costFn := func(i, j int) float64 { return float64((i-j)*(i-j)) * 0.1 }
	base := LogSinkhornConfig{Epsilon: 0.1, Tau1: 1.0, Tau2: 1.0, MaxIterations: 200, Tolerance: 1e-4}
	resBase, _ := accelCapture(t, M, N, logR, logC, costFn, base)
	for _, beta := range []float64{0.1, 0.3, 0.5} {
		cfg := base
		cfg.MomentumBeta = beta
		res, _ := accelCapture(t, M, N, logR, logC, costFn, cfg)
		if !res.Converged {
			t.Fatalf("beta=%v tidak konvergen", beta)
		}
		accelAssertFinite(t, "F", res.F)
		accelAssertFinite(t, "G", res.G)
		if math.Abs(resBase.Cost-res.Cost) > 1e-3 {
			t.Fatalf("beta=%v: cost %v menyimpang dari baseline %v", beta, res.Cost, resBase.Cost)
		}
	}
}

// [D] Sparse threshold di atas max cost = no-op: trajectory identik
// bitwise dengan run tanpa sparse (tidak ada entry yang ter-truncate).
func TestAcceleration_SparseNoOpAboveMaxCost(t *testing.T) {
	M, N := 5, 5
	logR := accelLogUniform(M)
	logC := accelLogUniform(N)
	base := LogSinkhornConfig{Epsilon: 0.1, Tau1: 1.0, Tau2: 1.0, MaxIterations: 50, Tolerance: 1e-4}
	_, seqPlain := accelCapture(t, M, N, logR, logC, accelBandCost, base)
	cfgSparse := base
	cfgSparse.SparseThreshold = 1000.0 // max cost = 100.0
	_, seqSparse := accelCapture(t, M, N, logR, logC, accelBandCost, cfgSparse)
	if d := accelSeqMaxDiff(seqPlain, seqSparse); d != 0 {
		t.Fatalf("threshold di atas max cost harus no-op (maxDiff=%e)", d)
	}
}

// [E] Sparse yang benar-benar memotong entry bermassa (exp(-c/eps)
// tidak underflow) harus mengubah trajectory secara signifikan.
func TestAcceleration_SparseTruncationChangesTrajectory(t *testing.T) {
	M, N := 5, 5
	logR := accelLogUniform(M)
	logC := accelLogUniform(N)
	base := LogSinkhornConfig{Epsilon: 0.1, Tau1: 1.0, Tau2: 1.0, MaxIterations: 200, Tolerance: 1e-4}
	resPlain, seqPlain := accelCapture(t, M, N, logR, logC, accelBandCost, base)
	cfgSparse := base
	cfgSparse.SparseThreshold = 0.15 // memangkas cost 0.2 (diff=2) & 100.0
	resSparse, seqSparse := accelCapture(t, M, N, logR, logC, accelBandCost, cfgSparse)
	if d := accelSeqMaxDiff(seqPlain, seqSparse); d < 1e-3 {
		t.Fatalf("truncation harus mengubah trajectory (maxDiff=%e)", d)
	}
	if !resPlain.Converged || !resSparse.Converged {
		t.Fatalf("kedua run harus konvergen (plain=%t, sparse=%t)", resPlain.Converged, resSparse.Converged)
	}
	accelAssertFinite(t, "F", resSparse.F)
	accelAssertFinite(t, "G", resSparse.G)
}

// [F] Logika sparse juga berlaku di cabang paralel (M,N >= 100).
func TestAcceleration_SparseParallelBranch(t *testing.T) {
	M, N := 128, 128
	logR := accelLogUniform(M)
	logC := accelLogUniform(N)
	cfg := LogSinkhornConfig{
		Epsilon:         0.1,
		Tau1:            1.0,
		Tau2:            1.0,
		MaxIterations:   30,
		Tolerance:       1e-4,
		SparseThreshold: 10.0,
	}
	res, err := LogSinkhornStreaming(M, N, logR, logC, accelBandCost, cfg)
	if err != nil {
		t.Fatalf("sparse di cabang paralel gagal: %v", err)
	}
	accelAssertFinite(t, "F", res.F)
	accelAssertFinite(t, "G", res.G)
	if math.IsNaN(res.Cost) || math.IsInf(res.Cost, 0) || res.Cost < 0 {
		t.Fatalf("cost tidak valid: %v", res.Cost)
	}
}

// [G] Kombinasi kedua fitur acceleration: momentum + sparse simultan
// tetap stabil dan konvergen.
func TestAcceleration_MomentumPlusSparseCombined(t *testing.T) {
	M, N := 5, 5
	logR := accelLogUniform(M)
	logC := accelLogUniform(N)
	cfg := LogSinkhornConfig{
		Epsilon:         0.1,
		Tau1:            1.0,
		Tau2:            1.0,
		MaxIterations:   200,
		Tolerance:       1e-4,
		SparseThreshold: 0.15,
		MomentumBeta:    0.2,
	}
	res, err := LogSinkhornStreaming(M, N, logR, logC, accelBandCost, cfg)
	if err != nil {
		t.Fatalf("kombinasi momentum+sparse gagal: %v", err)
	}
	if !res.Converged {
		t.Fatalf("kombinasi momentum+sparse tidak konvergen (residual %e)", res.FinalResidual)
	}
	accelAssertFinite(t, "F", res.F)
	accelAssertFinite(t, "G", res.G)
	if math.IsNaN(res.Cost) || math.IsInf(res.Cost, 0) || res.Cost < 0 {
		t.Fatalf("cost tidak valid: %v", res.Cost)
	}
}
