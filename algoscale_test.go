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

// ============================================================================
// PENDALAMAN: VARIAN EPSILON ANNEALING (LogSinkhornStreaming)
//
// Batasan domain (anti-overlap dengan file test lain):
// - TIDAK menguji MomentumBeta / SparseThreshold (domain acceleration/utilities)
// - TIDAK menguji ExportJSON/ImportJSON/Context/Observer (domain enterprise/observability)
// - TIDAK menguji kasus singular/extreme (domain penetration)
// - TIDAK menguji invariant umum algoritma (domain property/fuzz)
// Semua assertion berbasis invariant implementasi annealing itu sendiri:
// decay schedule bentuk tertutup, flag aktivasi, history, dan kualitas solusi.
// ============================================================================

// Helper berprefix "annealing" untuk menghindari bentrok identifier
// dengan file test lain dalam package yang sama.
func annealingLogUniform(n int) []float64 {
	out := make([]float64, n)
	v := -math.Log(float64(n))
	for i := range out {
		out[i] = v
	}
	return out
}

func annealingQuadCost(i, j int) float64 {
	d := float64(i - j)
	return d * d
}

func annealingAssertFinite(t *testing.T, label string, vals []float64) {
	t.Helper()
	for idx, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s[%d] tidak finite: %v", label, idx, v)
		}
	}
}

// [A] Decay schedule harus cocok persis dengan bentuk tertutup
// eps_t = Epsilon + (EpsilonInit-Epsilon)*exp(-5*t/MaxIterations)
func TestEpsilonAnnealingDecaySchedule(t *testing.T) {
	M, N := 5, 5
	cfg := LogSinkhornConfig{
		Epsilon:       0.01,
		EpsilonInit:   0.5,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 50,
		Tolerance:     1e-8, // sengaja ketat agar schedule berjalan penuh
	}
	var iters []int
	var epsSeq []float64
	cfg.OnIteration = func(iter int, residual float64, eps float64) {
		iters = append(iters, iter)
		epsSeq = append(epsSeq, eps)
	}
	if _, err := LogSinkhornStreaming(M, N, annealingLogUniform(M), annealingLogUniform(N), annealingQuadCost, cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(epsSeq) == 0 {
		t.Fatalf("OnIteration tidak pernah dipanggil")
	}
	// 1. eps pertama == EpsilonInit (decayFactor = 0)
	if math.Abs(epsSeq[0]-cfg.EpsilonInit) > 1e-12 {
		t.Fatalf("eps[0] = %v, ingin %v", epsSeq[0], cfg.EpsilonInit)
	}
	// 2. Cocok bentuk tertutup di setiap iterasi
	for idx, eps := range epsSeq {
		decay := float64(iters[idx]) / float64(cfg.MaxIterations)
		want := cfg.Epsilon + (cfg.EpsilonInit-cfg.Epsilon)*math.Exp(-5.0*decay)
		if math.Abs(eps-want) > 1e-12 {
			t.Fatalf("iter %d: eps=%v tidak sesuai schedule %v", iters[idx], eps, want)
		}
	}
	// 3. Monoton tidak naik dan selalu strict di atas Epsilon
	for i := 1; i < len(epsSeq); i++ {
		if epsSeq[i] > epsSeq[i-1]+1e-15 {
			t.Fatalf("eps naik di iter %d: %v -> %v", i, epsSeq[i-1], epsSeq[i])
		}
		if epsSeq[i] <= cfg.Epsilon {
			t.Fatalf("eps[%d]=%v menyentuh batas bawah %v", i, epsSeq[i], cfg.Epsilon)
		}
	}
}

// [B] Annealing harus NON-aktif bila EpsilonInit <= Epsilon (eps konstan).
func TestEpsilonAnnealingInactiveWhenInitNotGreater(t *testing.T) {
	cases := []struct {
		name    string
		epsInit float64
	}{
		{"InitNol", 0.0},
		{"InitLebihKecil", 0.05},
		{"InitSama", 0.1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := LogSinkhornConfig{
				Epsilon:       0.1,
				EpsilonInit:   tc.epsInit,
				Tau1:          1.0,
				Tau2:          1.0,
				MaxIterations: 10,
				Tolerance:     1e-12,
			}
			var epsSeq []float64
			cfg.OnIteration = func(_ int, _ float64, eps float64) {
				epsSeq = append(epsSeq, eps)
			}
			if _, err := LogSinkhornStreaming(4, 4, annealingLogUniform(4), annealingLogUniform(4), annealingQuadCost, cfg); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(epsSeq) == 0 {
				t.Fatalf("OnIteration tidak dipanggil")
			}
			for i, eps := range epsSeq {
				if eps != cfg.Epsilon {
					t.Fatalf("iter %d: annealing harus non-aktif, eps=%v ingin %v", i, eps, cfg.Epsilon)
				}
			}
		})
	}
}

// [C] Annealing + balanced (Tau=Inf) harus benar-benar konvergen,
// bukan sekadar bebas error; cost hasil harus valid.
func TestEpsilonAnnealingConvergesBalanced(t *testing.T) {
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		EpsilonInit:   0.5,
		Tau1:          math.Inf(1),
		Tau2:          math.Inf(1),
		MaxIterations: 500,
		Tolerance:     1e-4,
	}
	res, err := LogSinkhornStreaming(5, 5, annealingLogUniform(5), annealingLogUniform(5), annealingQuadCost, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Converged {
		t.Fatalf("annealing balanced tidak konvergen dalam %d iterasi (residual %e)", res.Iterations, res.FinalResidual)
	}
	annealingAssertFinite(t, "F", res.F)
	annealingAssertFinite(t, "G", res.G)
	if math.IsNaN(res.Cost) || math.IsInf(res.Cost, 0) || res.Cost < 0 {
		t.Fatalf("cost tidak valid: %v", res.Cost)
	}
}

// [D] Konsistensi internal: len(ResidualHistory) == Iterations dan
// elemen terakhir == FinalResidual.
func TestEpsilonAnnealingHistoryConsistency(t *testing.T) {
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		EpsilonInit:   0.5,
		Tau1:          math.Inf(1),
		Tau2:          math.Inf(1),
		MaxIterations: 300,
		Tolerance:     1e-4,
		EnableHistory: true,
	}
	res, err := LogSinkhornStreaming(5, 5, annealingLogUniform(5), annealingLogUniform(5), annealingQuadCost, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.ResidualHistory) != res.Iterations {
		t.Fatalf("panjang history %d != Iterations %d", len(res.ResidualHistory), res.Iterations)
	}
	if res.Iterations > 0 {
		last := res.ResidualHistory[len(res.ResidualHistory)-1]
		if last != res.FinalResidual {
			t.Fatalf("history terakhir %v != FinalResidual %v", last, res.FinalResidual)
		}
	}
}

// [E] Stabilitas numerik: target epsilon sangat kecil (0.005) tetap
// menghasilkan potensial & cost finite berkat annealing.
func TestEpsilonAnnealingStabilityTinyEpsilon(t *testing.T) {
	cfg := LogSinkhornConfig{
		Epsilon:       0.005,
		EpsilonInit:   0.5,
		Tau1:          math.Inf(1),
		Tau2:          math.Inf(1),
		MaxIterations: 150,
		Tolerance:     1e-6,
	}
	res, err := LogSinkhornStreaming(8, 8, annealingLogUniform(8), annealingLogUniform(8), annealingQuadCost, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	annealingAssertFinite(t, "F", res.F)
	annealingAssertFinite(t, "G", res.G)
	if math.IsNaN(res.Cost) || math.IsInf(res.Cost, 0) || res.Cost < 0 {
		t.Fatalf("cost tidak valid pada eps kecil: %v", res.Cost)
	}
}

// [F] Schedule annealing tetap berlaku di cabang paralel (M,N >= 100).
func TestEpsilonAnnealingParallelBranchLarge(t *testing.T) {
	M, N := 128, 128
	cost := func(i, j int) float64 {
		d := float64(i - j)
		return d * d * 0.01
	}
	cfg := LogSinkhornConfig{
		Epsilon:       0.2,
		EpsilonInit:   0.6,
		Tau1:          1.0,
		Tau2:          1.0,
		MaxIterations: 10,
		Tolerance:     1e-4,
	}
	var epsSeq []float64
	cfg.OnIteration = func(_ int, _ float64, eps float64) {
		epsSeq = append(epsSeq, eps)
	}
	res, err := LogSinkhornStreaming(M, N, annealingLogUniform(M), annealingLogUniform(N), cost, cfg)
	if err != nil {
		t.Fatalf("unexpected error di cabang paralel: %v", err)
	}
	if len(epsSeq) == 0 {
		t.Fatalf("OnIteration tidak dipanggil")
	}
	if math.Abs(epsSeq[0]-cfg.EpsilonInit) > 1e-12 {
		t.Fatalf("eps[0]=%v, ingin %v", epsSeq[0], cfg.EpsilonInit)
	}
	for i := 1; i < len(epsSeq); i++ {
		if epsSeq[i] > epsSeq[i-1]+1e-15 {
			t.Fatalf("eps naik di cabang paralel iter %d", i)
		}
	}
	annealingAssertFinite(t, "F", res.F)
	annealingAssertFinite(t, "G", res.G)
}

// [G] Kualitas solusi: plan dense hasil annealing (balanced) harus
// merekonstruksi marginal uniform setelah konvergen.
func TestEpsilonAnnealingDensePlanMarginals(t *testing.T) {
	M, N := 5, 5
	logR := annealingLogUniform(M)
	logC := annealingLogUniform(N)
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		EpsilonInit:   0.5,
		Tau1:          math.Inf(1),
		Tau2:          math.Inf(1),
		MaxIterations: 500,
		Tolerance:     1e-5,
	}
	lastEps := -1.0
	cfg.OnIteration = func(_ int, _ float64, eps float64) { lastEps = eps }
	res, err := LogSinkhornStreaming(M, N, logR, logC, annealingQuadCost, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Converged {
		t.Fatalf("tidak konvergen untuk validasi marginal (residual %e)", res.FinalResidual)
	}
	if lastEps <= 0 {
		t.Fatalf("eps terakhir tidak terekam")
	}
	P, err := res.ToDenseMatrix(M, N, logR, logC, annealingQuadCost, lastEps)
	if err != nil {
		t.Fatalf("ToDenseMatrix gagal: %v", err)
	}
	for i := 0; i < M; i++ {
		rowSum := 0.0
		for j := 0; j < N; j++ {
			rowSum += P.At(i, j)
		}
		if math.Abs(rowSum-1.0/float64(M)) > 1e-2 {
			t.Fatalf("marginal baris %d = %v, ingin ~%v", i, rowSum, 1.0/float64(M))
		}
	}
	for j := 0; j < N; j++ {
		colSum := 0.0
		for i := 0; i < M; i++ {
			colSum += P.At(i, j)
		}
		if math.Abs(colSum-1.0/float64(N)) > 1e-2 {
			t.Fatalf("marginal kolom %d = %v, ingin ~%v", j, colSum, 1.0/float64(N))
		}
	}
}
