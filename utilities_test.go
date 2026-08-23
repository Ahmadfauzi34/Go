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

// ============================================================================
// PENDALAMAN: DOMAIN UTILITIES (Stagnation Early-Stopping & ToDenseMatrix)
//
// Batasan domain (anti-overlap):
// - TIDAK menguji annealing/momentum/sparse (domain algoscale/acceleration)
// - TIDAK menguji JSON/observer/context, fuzz, property, penetration
// Semua test bebas annealing (EpsilonInit=0) sehingga invariant terhadap
// perubahan rescaling potensial pada engine.
// ============================================================================

// Helper berprefix "util" untuk menghindari bentrok identifier
// (file lain memakai prefix "annealing" dan "accel").
func utilLogUniform(n int) []float64 {
	out := make([]float64, n)
	v := -math.Log(float64(n))
	for i := range out {
		out[i] = v
	}
	return out
}

func utilQuadCost(i, j int) float64 {
	d := float64(i - j)
	return d * d
}

// [A] Semantik stagnation: berhenti SAAT penurunan residual dalam window
// < StagnationTol, men-set Converged, dan kondisi itu terlihat di history.
func TestUtilities_StagnationStopsOnSmallDecrease(t *testing.T) {
	M, N := 4, 4
	cfg := LogSinkhornConfig{
		Epsilon:          0.1,
		Tau1:             1.0,
		Tau2:             1.0,
		MaxIterations:    200,
		Tolerance:        1e-15,
		EnableHistory:    true,
		StagnationWindow: 3,
		StagnationTol:    1e-3,
	}
	res, err := LogSinkhornStreaming(M, N, utilLogUniform(M), utilLogUniform(N), utilQuadCost, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming gagal: %v", err)
	}
	if !res.Converged {
		t.Fatalf("stagnation harus men-set flag Converged")
	}
	if res.Iterations >= cfg.MaxIterations {
		t.Fatalf("stagnation harus berhenti sebelum MaxIterations, got %d", res.Iterations)
	}
	if len(res.ResidualHistory) != res.Iterations {
		t.Fatalf("panjang history %d != Iterations %d", len(res.ResidualHistory), res.Iterations)
	}
	L := len(res.ResidualHistory)
	past := res.ResidualHistory[L-cfg.StagnationWindow]
	final := res.ResidualHistory[L-1]
	if past-final >= cfg.StagnationTol {
		t.Fatalf("syarat stagnasi tidak terlihat di history: past=%e final=%e", past, final)
	}
	if res.FinalResidual != final {
		t.Fatalf("FinalResidual %e != elemen history terakhir %e", res.FinalResidual, final)
	}
	if res.FinalResidual <= cfg.Tolerance {
		t.Fatalf("berhenti via tolerance, bukan via stagnation")
	}
}

// [B] Window=1: pastRes == residual saat ini → selisih 0 < stTol →
// berhenti tepat di iterasi pertama (structural).
func TestUtilities_StagnationImmediateStopWindowOne(t *testing.T) {
	cfg := LogSinkhornConfig{
		Epsilon:          0.1,
		Tau1:             1.0,
		Tau2:             1.0,
		MaxIterations:    50,
		Tolerance:        1e-15,
		StagnationWindow: 1,
		StagnationTol:    1e-1,
	}
	res, err := LogSinkhornStreaming(4, 4, utilLogUniform(4), utilLogUniform(4), utilQuadCost, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming gagal: %v", err)
	}
	if res.Iterations != 1 {
		t.Fatalf("window=1 harus berhenti di iterasi pertama, got %d", res.Iterations)
	}
	if !res.Converged {
		t.Fatalf("stagnation window=1 harus men-set Converged")
	}
	if len(res.ResidualHistory) != 1 {
		t.Fatalf("history harus berisi 1 entri, got %d", len(res.ResidualHistory))
	}
}

// [C] Window=0 → stagnation disabled → jalan penuh sampai MaxIterations.
func TestUtilities_StagnationDisabledByZeroWindow(t *testing.T) {
	cfg := LogSinkhornConfig{
		Epsilon:          0.1,
		Tau1:             1.0,
		Tau2:             1.0,
		MaxIterations:    30,
		Tolerance:        1e-15,
		StagnationWindow: 0,
		StagnationTol:    1e-3,
	}
	res, err := LogSinkhornStreaming(4, 4, utilLogUniform(4), utilLogUniform(4), utilQuadCost, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming gagal: %v", err)
	}
	if res.Iterations != 30 {
		t.Fatalf("tanpa window harus penuh 30 iterasi, got %d", res.Iterations)
	}
	if res.Converged {
		t.Fatalf("tidak boleh converged saat tolerance tak terjangkau")
	}
	if len(res.ResidualHistory) != 0 {
		t.Fatalf("history harus kosong, got %d", len(res.ResidualHistory))
	}
}

// [D] StagnationTol <= 0 → guard stTol > 0 menonaktifkan early stopping.
func TestUtilities_StagnationDisabledByNonPositiveTol(t *testing.T) {
	for _, stTol := range []float64{0.0, -1.0} {
		cfg := LogSinkhornConfig{
			Epsilon:          0.1,
			Tau1:             1.0,
			Tau2:             1.0,
			MaxIterations:    30,
			Tolerance:        1e-15,
			StagnationWindow: 3,
			StagnationTol:    stTol,
		}
		res, err := LogSinkhornStreaming(4, 4, utilLogUniform(4), utilLogUniform(4), utilQuadCost, cfg)
		if err != nil {
			t.Fatalf("stTol=%v: LogSinkhornStreaming gagal: %v", stTol, err)
		}
		if res.Iterations != 30 || res.Converged {
			t.Fatalf("stTol=%v: stagnation harus non-aktif (iter=%d, converged=%t)", stTol, res.Iterations, res.Converged)
		}
	}
}

// [E] Window > MaxIterations: kondisi len(history) >= w tak pernah
// terpenuhi; history tetap diisi karena window > 0 (walau EnableHistory false).
func TestUtilities_StagnationWindowExceedsMaxIterations(t *testing.T) {
	cfg := LogSinkhornConfig{
		Epsilon:          0.1,
		Tau1:             1.0,
		Tau2:             1.0,
		MaxIterations:    30,
		Tolerance:        1e-15,
		StagnationWindow: 1000,
		StagnationTol:    1e-3,
	}
	res, err := LogSinkhornStreaming(4, 4, utilLogUniform(4), utilLogUniform(4), utilQuadCost, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming gagal: %v", err)
	}
	if res.Iterations != 30 || res.Converged {
		t.Fatalf("window raksasa tidak boleh memicu stop (iter=%d, converged=%t)", res.Iterations, res.Converged)
	}
	if len(res.ResidualHistory) != 30 {
		t.Fatalf("history tetap diisi saat window>0: got %d, ingin 30", len(res.ResidualHistory))
	}
}

// [F] Cabang error ToDenseMatrix: eps non-positif, mismatch dimensi, receiver nil.
func TestUtilities_ToDenseMatrixErrorPaths(t *testing.T) {
	M, N := 2, 2
	logR := utilLogUniform(M)
	logC := utilLogUniform(N)
	cfg := LogSinkhornConfig{Epsilon: 0.1, Tau1: 1.0, Tau2: 1.0, MaxIterations: 50, Tolerance: 1e-4}
	res, err := LogSinkhornStreaming(M, N, logR, logC, utilQuadCost, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming gagal: %v", err)
	}
	if _, e := res.ToDenseMatrix(M, N, logR, logC, utilQuadCost, 0.0); e == nil {
		t.Fatalf("eps=0 harus error")
	}
	if _, e := res.ToDenseMatrix(M, N, logR, logC, utilQuadCost, -0.5); e == nil {
		t.Fatalf("eps negatif harus error")
	}
	if _, e := res.ToDenseMatrix(M+1, N, logR, logC, utilQuadCost, 0.1); e == nil {
		t.Fatalf("mismatch M harus error")
	}
	if _, e := res.ToDenseMatrix(M, N+1, logR, logC, utilQuadCost, 0.1); e == nil {
		t.Fatalf("mismatch N harus error")
	}
	var nilRes *LogSinkhornResult
	if _, e := nilRes.ToDenseMatrix(M, N, logR, logC, utilQuadCost, 0.1); e == nil {
		t.Fatalf("receiver nil harus error")
	}
}

// [G] Konsistensi struktural: Σ P_ij·c_ij dari ToDenseMatrix (tanpa
// annealing → eps konstan) harus mereproduksi res.Cost dari engine.
func TestUtilities_ToDenseMatrixCostConsistency(t *testing.T) {
	M, N := 5, 5
	logR := utilLogUniform(M)
	logC := utilLogUniform(N)
	cfg := LogSinkhornConfig{Epsilon: 0.1, Tau1: 1.0, Tau2: 1.0, MaxIterations: 200, Tolerance: 1e-6}
	res, err := LogSinkhornStreaming(M, N, logR, logC, utilQuadCost, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming gagal: %v", err)
	}
	P, errP := res.ToDenseMatrix(M, N, logR, logC, utilQuadCost, cfg.Epsilon)
	if errP != nil {
		t.Fatalf("ToDenseMatrix gagal: %v", errP)
	}
	recomputed := 0.0
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			p := P.At(i, j)
			if math.IsNaN(p) || math.IsInf(p, 0) {
				t.Fatalf("P[%d,%d] tidak finite: %v", i, j, p)
			}
			if p <= 0 {
				t.Fatalf("P[%d,%d] harus positif (exp), got %v", i, j, p)
			}
			recomputed += p * utilQuadCost(i, j)
		}
	}
	if math.Abs(recomputed-res.Cost) > 1e-9*(1+math.Abs(res.Cost)) {
		t.Fatalf("cost rekonstruksi %v != Cost engine %v", recomputed, res.Cost)
	}
}

// [H] Massa total plan dense ≈ 1 pada run balanced (Tau=Inf) yang konvergen.
func TestUtilities_ToDenseMatrixTotalMassBalanced(t *testing.T) {
	M, N := 4, 4
	logR := utilLogUniform(M)
	logC := utilLogUniform(N)
	cfg := LogSinkhornConfig{
		Epsilon:       0.1,
		Tau1:          math.Inf(1),
		Tau2:          math.Inf(1),
		MaxIterations: 300,
		Tolerance:     1e-5,
	}
	res, err := LogSinkhornStreaming(M, N, logR, logC, utilQuadCost, cfg)
	if err != nil {
		t.Fatalf("LogSinkhornStreaming gagal: %v", err)
	}
	if !res.Converged {
		t.Fatalf("run balanced harus konvergen (residual %e)", res.FinalResidual)
	}
	P, errP := res.ToDenseMatrix(M, N, logR, logC, utilQuadCost, cfg.Epsilon)
	if errP != nil {
		t.Fatalf("ToDenseMatrix gagal: %v", errP)
	}
	total := 0.0
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			total += P.At(i, j)
		}
	}
	if math.Abs(total-1.0) > 1e-2 {
		t.Fatalf("massa total plan %v, ingin ≈ 1.0", total)
	}
}
