package algoscale

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// ============================================================================
// 1. MEMORY ABSTRACTION & STABLE MATRIX INVERSION
// ============================================================================

type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

func NewMatrix(rows, cols int) *Matrix {
	if rows <= 0 || cols <= 0 {
		panic("dimensi matriks harus bernilai positif (> 0)")
	}
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float64, rows*cols),
	}
}

func (m *Matrix) At(r, c int) float64  { return m.Data[r*m.Cols+c] }
func (m *Matrix) Set(r, c int, v float64) { m.Data[r*m.Cols+c] = v }

// InvertSmallMatrix menginversi matriks persegi r x r via Gauss-Jordan
// dengan Dynamic / Relative Pivot Thresholding.
func InvertSmallMatrix(A *Matrix) (*Matrix, error) {
	if A == nil {
		return nil, errors.New("matriks A tidak boleh bernilai nil")
	}
	if A.Rows != A.Cols {
		return nil, errors.New("matriks harus persegi untuk diinversi")
	}
	n := A.Rows
	if n == 0 {
		return nil, errors.New("dimensi matriks tidak boleh kosong")
	}

	normInf := 0.0
	for i := 0; i < n; i++ {
		rowSum := 0.0
		offset := i * n
		for j := 0; j < n; j++ {
			rowSum += math.Abs(A.Data[offset+j])
		}
		if rowSum > normInf {
			normInf = rowSum
		}
	}
	if normInf == 0.0 {
		return nil, errors.New("matriks bernilai nol mutlak (singular)")
	}

	const epsMach = 2.220446049250313e-16
	pivotTolerance := math.Max(1e-15, float64(n)*epsMach*normInf)

	aug := make([]float64, n*2*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug[i*(2*n)+j] = A.Data[i*n+j]
		}
		aug[i*(2*n)+n+i] = 1.0
	}

	for i := 0; i < n; i++ {
		pivotRow := i
		maxVal := math.Abs(aug[i*(2*n)+i])
		for k := i + 1; k < n; k++ {
			if val := math.Abs(aug[k*(2*n)+i]); val > maxVal {
				maxVal = val
				pivotRow = k
			}
		}

		if maxVal < pivotTolerance {
			return nil, fmt.Errorf("matriks singular secara numerik (pivot %e < toleransi %e)", maxVal, pivotTolerance)
		}

		if pivotRow != i {
			for j := 0; j < 2*n; j++ {
				aug[i*(2*n)+j], aug[pivotRow*(2*n)+j] = aug[pivotRow*(2*n)+j], aug[i*(2*n)+j]
			}
		}

		pivot := aug[i*(2*n)+i]
		for j := 0; j < 2*n; j++ {
			aug[i*(2*n)+j] /= pivot
		}

		for k := 0; k < n; k++ {
			if k != i {
				factor := aug[k*(2*n)+i]
				for j := 0; j < 2*n; j++ {
					aug[k*(2*n)+j] -= factor * aug[i*(2*n)+j]
				}
			}
		}
	}

	inv := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			inv.Data[i*n+j] = aug[i*(2*n)+n+j]
		}
	}
	return inv, nil
}

// ============================================================================
// 2. NUMERICAL LOG-SUM-EXP PRIMITIVES
// ============================================================================

func LogSumExp(vals []float64) float64 {
	if len(vals) == 0 {
		return math.Inf(-1)
	}
	maxVal := vals[0]
	for _, v := range vals[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	if math.IsInf(maxVal, -1) {
		return math.Inf(-1)
	}

	sumExp := 0.0
	for _, v := range vals {
		sumExp += math.Exp(v - maxVal)
	}
	return maxVal + math.Log(sumExp)
}

func LogSumExpWeighted(vals []float64, logW []float64) float64 {
	if len(vals) == 0 || len(vals) != len(logW) {
		return math.Inf(-1)
	}
	maxVal := math.Inf(-1)
	for j := range vals {
		term := vals[j] + logW[j]
		if term > maxVal {
			maxVal = term
		}
	}
	if math.IsInf(maxVal, -1) {
		return math.Inf(-1)
	}

	sumExp := 0.0
	for j := range vals {
		sumExp += math.Exp((vals[j] + logW[j]) - maxVal)
	}
	return maxVal + math.Log(sumExp)
}

// ============================================================================
// 3. UNBALANCED SINKHORN (DENSE PRIMAL)
// ============================================================================

type UnbalancedConfig struct {
	Epsilon       float64
	Tau1          float64
	Tau2          float64
	MaxIterations int
	Tolerance     float64
}

type UnbalancedResult struct {
	Plan           *Matrix
	U              []float64
	V              []float64
	Iterations     int
	FinalResidual  float64
	Converged      bool
	MarginalError1 float64
	MarginalError2 float64
}

func UnbalancedSinkhorn(K *Matrix, r, c []float64, cfg UnbalancedConfig) (*UnbalancedResult, error) {
	if K == nil {
		return nil, errors.New("matriks kernel K tidak boleh nil")
	}
	m, n := K.Rows, K.Cols
	if len(r) != m || len(c) != n {
		return nil, fmt.Errorf("dimensi marginal tidak cocok: len(r)=%d vs K.Rows=%d, len(c)=%d vs K.Cols=%d", len(r), m, len(c), n)
	}
	if cfg.Epsilon <= 0 || cfg.Tau1 <= 0 || cfg.Tau2 <= 0 {
		return nil, errors.New("hiperparameter Epsilon, Tau1, dan Tau2 harus bernilai riil positif (> 0)")
	}
	if cfg.MaxIterations <= 0 || cfg.Tolerance <= 0 {
		return nil, errors.New("MaxIterations dan Tolerance harus bernilai positif (> 0)")
	}

	for i, val := range r {
		if val < 0 || math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("marginal r[%d] tidak valid: %v", i, val)
		}
	}
	for j, val := range c {
		if val < 0 || math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("marginal c[%d] tidak valid: %v", j, val)
		}
	}

	kappa1 := cfg.Tau1 / (cfg.Tau1 + cfg.Epsilon)
	kappa2 := cfg.Tau2 / (cfg.Tau2 + cfg.Epsilon)

	u := make([]float64, m)
	v := make([]float64, n)
	uPrev := make([]float64, m)
	vPrev := make([]float64, n)
	for i := range u { u[i] = 1.0 }
	for j := range v { v[j] = 1.0 }

	tempKv := make([]float64, m)
	tempKtu := make([]float64, n)

	var iter int
	var residual float64
	var converged bool

	for iter = 0; iter < cfg.MaxIterations; iter++ {
		copy(uPrev, u)
		copy(vPrev, v)

		for i := 0; i < m; i++ {
			sum := 0.0
			offset := i * n
			for j := 0; j < n; j++ {
				sum += K.Data[offset+j] * v[j]
			}
			tempKv[i] = sum
		}

		for i := 0; i < m; i++ {
			if tempKv[i] > 1e-300 {
				u[i] = math.Pow(r[i]/tempKv[i], kappa1)
			} else {
				u[i] = 0.0
			}
		}

		for j := 0; j < n; j++ {
			tempKtu[j] = 0.0
		}
		for i := 0; i < m; i++ {
			ui := u[i]
			offset := i * n
			for j := 0; j < n; j++ {
				tempKtu[j] += K.Data[offset+j] * ui
			}
		}

		for j := 0; j < n; j++ {
			if tempKtu[j] > 1e-300 {
				v[j] = math.Pow(c[j]/tempKtu[j], kappa2)
			} else {
				v[j] = 0.0
			}
		}

		resU := 0.0
		for i := 0; i < m; i++ {
			diff := math.Abs(u[i] - uPrev[i])
			if diff > resU { resU = diff }
		}
		resV := 0.0
		for j := 0; j < n; j++ {
			diff := math.Abs(v[j] - vPrev[j])
			if diff > resV { resV = diff }
		}

		residual = math.Max(resU, resV)
		if residual < cfg.Tolerance {
			converged = true
			iter++
			break
		}
	}

	P := NewMatrix(m, n)
	for i := 0; i < m; i++ {
		ui := u[i]
		offset := i * n
		for j := 0; j < n; j++ {
			P.Data[offset+j] = ui * K.Data[offset+j] * v[j]
		}
	}

	var err1, err2 float64
	for i := 0; i < m; i++ {
		rowSum := 0.0
		offset := i * n
		for j := 0; j < n; j++ { rowSum += P.Data[offset+j] }
		err1 += math.Abs(rowSum - r[i])
	}
	for j := 0; j < n; j++ {
		colSum := 0.0
		for i := 0; i < m; i++ { colSum += P.Data[i*n+j] }
		err2 += math.Abs(colSum - c[j])
	}

	return &UnbalancedResult{
		Plan:           P,
		U:              u,
		V:              v,
		Iterations:     iter,
		FinalResidual:  residual,
		Converged:      converged,
		MarginalError1: err1,
		MarginalError2: err2,
	}, nil
}

// ============================================================================
// 4. LOW-RANK NYSTROM FACTORED SINKHORN
// ============================================================================

type LowRankKernel struct {
	M      int
	N      int
	Rank   int
	Kxz    *Matrix
	KzzInv *Matrix
	Kzy    *Matrix
}

func NewLowRankKernel(M, N, Rank int, Kxz, Kzz, Kzy *Matrix) (*LowRankKernel, error) {
	if Kxz == nil || Kzz == nil || Kzy == nil {
		return nil, errors.New("matriks input (Kxz, Kzz, Kzy) tidak boleh bernilai nil")
	}
	if M <= 0 || N <= 0 || Rank <= 0 {
		return nil, errors.New("dimensi M, N, dan Rank harus bernilai positif (> 0)")
	}
	if Kxz.Rows != M || Kxz.Cols != Rank {
		return nil, fmt.Errorf("dimensi Kxz tidak valid: (%d, %d), ekspektasi (%d, %d)", Kxz.Rows, Kxz.Cols, M, Rank)
	}
	if Kzy.Rows != Rank || Kzy.Cols != N {
		return nil, fmt.Errorf("dimensi Kzy tidak valid: (%d, %d), ekspektasi (%d, %d)", Kzy.Rows, Kzy.Cols, Rank, N)
	}
	if Kzz.Rows != Rank || Kzz.Cols != Rank {
		return nil, fmt.Errorf("dimensi Kzz tidak valid: (%d, %d), ekspektasi (%d, %d)", Kzz.Rows, Kzz.Cols, Rank, Rank)
	}

	kzzInv, err := InvertSmallMatrix(Kzz)
	if err != nil {
		return nil, fmt.Errorf("inversi Kzz gagal: %w", err)
	}

	return &LowRankKernel{
		M:      M,
		N:      N,
		Rank:   Rank,
		Kxz:    Kxz,
		KzzInv: kzzInv,
		Kzy:    Kzy,
	}, nil
}

type LowRankResult struct {
	U             []float64
	V             []float64
	Iterations    int
	FinalResidual float64
	Converged     bool
}

func (lrk *LowRankKernel) Solve(r, c []float64, maxIter int, tol float64) (*LowRankResult, error) {
	if lrk == nil {
		return nil, errors.New("receiver LowRankKernel tidak boleh nil")
	}
	m, n, rank := lrk.M, lrk.N, lrk.Rank

	if len(r) != m || len(c) != n {
		return nil, fmt.Errorf("dimensi marginal tidak cocok: len(r)=%d (ekspektasi %d), len(c)=%d (ekspektasi %d)", len(r), m, len(c), n)
	}
	if maxIter <= 0 || tol <= 0 {
		return nil, errors.New("maxIter dan tol harus bernilai positif (> 0)")
	}

	for i, val := range r {
		if val < 0 || math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("marginal r[%d] tidak valid: %v", i, val)
		}
	}
	for j, val := range c {
		if val < 0 || math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("marginal c[%d] tidak valid: %v", j, val)
		}
	}

	u := make([]float64, m)
	v := make([]float64, n)
	uPrev := make([]float64, m)
	vPrev := make([]float64, n)
	for i := range u { u[i] = 1.0 }
	for j := range v { v[j] = 1.0 }

	t1 := make([]float64, rank)
	t2 := make([]float64, rank)
	kv := make([]float64, m)

	s1 := make([]float64, rank)
	s2 := make([]float64, rank)
	ktu := make([]float64, n)

	var residual float64
	var converged bool
	var iter int

	for iter = 0; iter < maxIter; iter++ {
		copy(uPrev, u)
		copy(vPrev, v)

		for k := 0; k < rank; k++ {
			sum := 0.0
			offset := k * n
			for j := 0; j < n; j++ { sum += lrk.Kzy.Data[offset+j] * v[j] }
			t1[k] = sum
		}

		for k := 0; k < rank; k++ {
			sum := 0.0
			offset := k * rank
			for l := 0; l < rank; l++ { sum += lrk.KzzInv.Data[offset+l] * t1[l] }
			t2[k] = sum
		}

		for i := 0; i < m; i++ {
			sum := 0.0
			offset := i * rank
			for k := 0; k < rank; k++ { sum += lrk.Kxz.Data[offset+k] * t2[k] }
			kv[i] = sum
		}

		for i := 0; i < m; i++ {
			if kv[i] > 1e-300 {
				u[i] = r[i] / kv[i]
			} else {
				u[i] = 0.0
			}
		}

		for k := 0; k < rank; k++ { s1[k] = 0.0 }
		for i := 0; i < m; i++ {
			ui := u[i]
			offset := i * rank
			for k := 0; k < rank; k++ { s1[k] += lrk.Kxz.Data[offset+k] * ui }
		}

		for l := 0; l < rank; l++ { s2[l] = 0.0 }
		for k := 0; k < rank; k++ {
			sk := s1[k]
			offset := k * rank
			for l := 0; l < rank; l++ { s2[l] += lrk.KzzInv.Data[offset+l] * sk }
		}

		for j := 0; j < n; j++ { ktu[j] = 0.0 }
		for k := 0; k < rank; k++ {
			sk := s2[k]
			offset := k * n
			for j := 0; j < n; j++ { ktu[j] += lrk.Kzy.Data[offset+j] * sk }
		}

		for j := 0; j < n; j++ {
			if ktu[j] > 1e-300 {
				v[j] = c[j] / ktu[j]
			} else {
				v[j] = 0.0
			}
		}

		resU := 0.0
		for i := 0; i < m; i++ {
			diff := math.Abs(u[i] - uPrev[i])
			if diff > resU { resU = diff }
		}
		resV := 0.0
		for j := 0; j < n; j++ {
			diff := math.Abs(v[j] - vPrev[j])
			if diff > resV { resV = diff }
		}

		residual = math.Max(resU, resV)
		if residual < tol {
			converged = true
			iter++
			break
		}
	}

	return &LowRankResult{
		U:             u,
		V:             v,
		Iterations:    iter,
		FinalResidual: residual,
		Converged:     converged,
	}, nil
}

// ============================================================================
// 5. MATRIX-FREE LOG-DOMAIN SINKHORN ENGINE & DEBIASED DIVERGENCE
// ============================================================================

type LogSinkhornConfig struct {
	Epsilon       float64
	EpsilonInit   float64 // If > Epsilon, enables exponential Epsilon Annealing
	Tau1          float64
	Tau2          float64
	MaxIterations int
	Tolerance     float64
	EnableHistory bool
	OnIteration   func(iter int, residual float64, eps float64)
}

type LogSinkhornResult struct {
	F               []float64
	G               []float64
	Iterations      int
	FinalResidual   float64
	ResidualHistory []float64
	Converged       bool
	Cost            float64
}

type CostFunction func(i, j int) float64

func LogSinkhornStreaming(
	M, N int,
	logR, logC []float64,
	costFn CostFunction,
	cfg LogSinkhornConfig,
) (*LogSinkhornResult, error) {
	if M <= 0 || N <= 0 {
		return nil, errors.New("dimensi M dan N harus bernilai positif")
	}
	if len(logR) != M || len(logC) != N {
		return nil, fmt.Errorf("dimensi log-marginal mismatch: len(logR)=%d vs %d, len(logC)=%d vs %d", len(logR), M, len(logC), N)
	}
	if cfg.Epsilon <= 0 || cfg.MaxIterations <= 0 || cfg.Tolerance <= 0 {
		return nil, errors.New("konfigurasi tidak valid: Epsilon, MaxIterations, dan Tolerance harus > 0")
	}
	if cfg.Tau1 <= 0 || cfg.Tau2 <= 0 {
		return nil, errors.New("konfigurasi tidak valid: Tau1 dan Tau2 harus > 0")
	}

	eps := cfg.Epsilon
	invEps := 1.0 / eps

	kappa1 := cfg.Tau1 / (cfg.Tau1 + eps)
	kappa2 := cfg.Tau2 / (cfg.Tau2 + eps)
	if math.IsInf(cfg.Tau1, 1) { kappa1 = 1.0 }
	if math.IsInf(cfg.Tau2, 1) { kappa2 = 1.0 }

	f := make([]float64, M)
	g := make([]float64, N)
	fPrev := make([]float64, M)
	gPrev := make([]float64, N)
	buffer := make([]float64, int(math.Max(float64(M), float64(N))))

	var resHistory []float64
	if cfg.EnableHistory {
		resHistory = make([]float64, 0, cfg.MaxIterations)
	}

	var iter int
	var residual float64
	var converged bool

	epsInit := cfg.EpsilonInit
	useAnnealing := epsInit > cfg.Epsilon

	for iter = 0; iter < cfg.MaxIterations; iter++ {
		if useAnnealing {
			// Exponential decay from epsInit down to cfg.Epsilon over MaxIterations
			decayFactor := float64(iter) / float64(cfg.MaxIterations)
			eps = cfg.Epsilon + (epsInit-cfg.Epsilon)*math.Exp(-5.0*decayFactor)
			invEps = 1.0 / eps
			kappa1 = cfg.Tau1 / (cfg.Tau1 + eps)
			kappa2 = cfg.Tau2 / (cfg.Tau2 + eps)
			if math.IsInf(cfg.Tau1, 1) { kappa1 = 1.0 }
			if math.IsInf(cfg.Tau2, 1) { kappa2 = 1.0 }
		}

		copy(fPrev, f)
		copy(gPrev, g)

		numWorkers := runtime.GOMAXPROCS(0)
		if numWorkers < 1 {
			numWorkers = 1
		}

		if M >= 100 || N >= 100 {
			var wg sync.WaitGroup

			chunkSizeM := (M + numWorkers - 1) / numWorkers
			for w := 0; w < numWorkers; w++ {
				start := w * chunkSizeM
				end := start + chunkSizeM
				if start >= M {
					break
				}
				if end > M {
					end = M
				}
				wg.Add(1)
				go func(startI, endI int) {
					defer wg.Done()
					buf := make([]float64, N)
					for i := startI; i < endI; i++ {
						for j := 0; j < N; j++ {
							buf[j] = (g[j] - costFn(i, j)) * invEps
						}
						lse := LogSumExpWeighted(buf, logC)
						f[i] = kappa1 * (-eps*lse + eps*logR[i])
					}
				}(start, end)
			}
			wg.Wait()

			chunkSizeN := (N + numWorkers - 1) / numWorkers
			for w := 0; w < numWorkers; w++ {
				start := w * chunkSizeN
				end := start + chunkSizeN
				if start >= N {
					break
				}
				if end > N {
					end = N
				}
				wg.Add(1)
				go func(startJ, endJ int) {
					defer wg.Done()
					buf := make([]float64, M)
					for j := startJ; j < endJ; j++ {
						for i := 0; i < M; i++ {
							buf[i] = (f[i] - costFn(i, j)) * invEps
						}
						lse := LogSumExpWeighted(buf, logR)
						g[j] = kappa2 * (-eps*lse + eps*logC[j])
					}
				}(start, end)
			}
			wg.Wait()
		} else {
			for i := 0; i < M; i++ {
				for j := 0; j < N; j++ {
					c_ij := costFn(i, j)
					buffer[j] = (g[j] - c_ij) * invEps
				}
				lse := LogSumExpWeighted(buffer[:N], logC)
				f[i] = kappa1 * (-eps*lse + eps*logR[i])
			}

			for j := 0; j < N; j++ {
				for i := 0; i < M; i++ {
					c_ij := costFn(i, j)
					buffer[i] = (f[i] - c_ij) * invEps
				}
				lse := LogSumExpWeighted(buffer[:M], logR)
				g[j] = kappa2 * (-eps*lse + eps*logC[j])
			}
		}

		resF := 0.0
		for i := 0; i < M; i++ {
			diff := math.Abs(f[i] - fPrev[i])
			if diff > resF { resF = diff }
		}
		resG := 0.0
		for j := 0; j < N; j++ {
			diff := math.Abs(g[j] - gPrev[j])
			if diff > resG { resG = diff }
		}

		residual = math.Max(resF, resG)
		if cfg.EnableHistory {
			resHistory = append(resHistory, residual)
		}
		if cfg.OnIteration != nil {
			cfg.OnIteration(iter, residual, eps)
		}

		if residual < cfg.Tolerance {
			converged = true
			iter++
			break
		}
	}

	totalCost := 0.0
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			c_ij := costFn(i, j)
			logP_ij := (f[i] + g[j] - c_ij)*invEps + logR[i] + logC[j]
			p_ij := math.Exp(logP_ij)
			totalCost += p_ij * c_ij
		}
	}

	return &LogSinkhornResult{
		F:               f,
		G:               g,
		Iterations:      iter,
		FinalResidual:   residual,
		ResidualHistory: resHistory,
		Converged:       converged,
		Cost:            totalCost,
	}, nil
}

// SinkhornDivergence menghitung metrik jarak debiased S_eps(X, Y)
func SinkhornDivergence(
	X, Y [][]float64,
	costMetric func(a, b []float64) float64,
	cfg LogSinkhornConfig,
) (float64, error) {
	M := len(X)
	N := len(Y)
	if M == 0 || N == 0 {
		return 0, errors.New("point clouds tidak boleh kosong")
	}

	logR := make([]float64, M)
	for i := range logR { logR[i] = -math.Log(float64(M)) }
	logC := make([]float64, N)
	for j := range logC { logC[j] = -math.Log(float64(N)) }

	costXY := func(i, j int) float64 { return costMetric(X[i], Y[j]) }
	costXX := func(i, j int) float64 { return costMetric(X[i], X[j]) }
	costYY := func(i, j int) float64 { return costMetric(Y[i], Y[j]) }

	cfgBalanced := cfg
	cfgBalanced.Tau1 = math.Inf(1)
	cfgBalanced.Tau2 = math.Inf(1)

	resXY, err := LogSinkhornStreaming(M, N, logR, logC, costXY, cfgBalanced)
	if err != nil { return 0, err }
	resXX, err := LogSinkhornStreaming(M, M, logR, logR, costXX, cfgBalanced)
	if err != nil { return 0, err }
	resYY, err := LogSinkhornStreaming(N, N, logC, logC, costYY, cfgBalanced)
	if err != nil { return 0, err }

	div := resXY.Cost - 0.5*resXX.Cost - 0.5*resYY.Cost
	if div < 0 && div > -1e-12 { div = 0.0 }
	return div, nil
}

// ============================================================================
// 6. SINKHORN BARYCENTERS (LOG-DOMAIN)
// ============================================================================

type BarycenterConfig struct {
	Epsilon       float64
	MaxIterations int
	Tolerance     float64
}

type BarycenterResult struct {
	Barycenter    []float64
	LogBarycenter []float64
	Iterations    int
	FinalResidual float64
	Converged     bool
	Subcosts      []float64
	WeightedCost  float64
}

func ComputeBarycenterLogDomain(
	distributions [][]float64,
	weights []float64,
	costFn func(i, j int) float64,
	M, N int,
	cfg BarycenterConfig,
) (*BarycenterResult, error) {
	K := len(distributions)
	if K == 0 {
		return nil, errors.New("tidak ada distribusi input yang diberikan")
	}
	if len(weights) != K {
		return nil, fmt.Errorf("jumlah bobot (%d) tidak cocok dengan jumlah distribusi (%d)", len(weights), K)
	}
	if M <= 0 || N <= 0 {
		return nil, errors.New("dimensi grid M dan N harus bernilai positif")
	}
	if cfg.Epsilon <= 0 || cfg.MaxIterations <= 0 || cfg.Tolerance <= 0 {
		return nil, errors.New("konfigurasi tidak valid: Epsilon, MaxIterations, dan Tolerance harus > 0")
	}

	sumLambda := 0.0
	for k, w := range weights {
		if w < 0 || math.IsNaN(w) || math.IsInf(w, 0) {
			return nil, fmt.Errorf("bobot lambda[%d] tidak valid: %v", k, w)
		}
		sumLambda += w
	}
	if math.Abs(sumLambda-1.0) > 1e-5 {
		return nil, fmt.Errorf("total bobot lambda harus bernilai 1.0 (didapat: %f)", sumLambda)
	}

	logP := make([][]float64, K)
	for k := 0; k < K; k++ {
		if len(distributions[k]) != N {
			return nil, fmt.Errorf("panjang distribusi[%d] (%d) != N (%d)", k, len(distributions[k]), N)
		}
		logP[k] = make([]float64, N)
		sumPk := 0.0
		for _, val := range distributions[k] {
			if val < 0 || math.IsNaN(val) || math.IsInf(val, 0) {
				return nil, fmt.Errorf("distribusi[%d] mengandung nilai negatif/NaN", k)
			}
			sumPk += val
		}
		if sumPk <= 0 {
			return nil, fmt.Errorf("distribusi[%d] bernilai nol mutlak", k)
		}
		for j, val := range distributions[k] {
			normVal := val / sumPk
			if normVal > 1e-300 {
				logP[k][j] = math.Log(normVal)
			} else {
				logP[k][j] = -700.0
			}
		}
	}

	eps := cfg.Epsilon
	invEps := 1.0 / eps

	f := make([][]float64, K)
	g := make([][]float64, K)
	for k := 0; k < K; k++ {
		f[k] = make([]float64, M)
		g[k] = make([]float64, N)
	}

	logQ := make([]float64, M)
	logQPrev := make([]float64, M)
	initLogQ := -math.Log(float64(M))
	for i := range logQ { logQ[i] = initLogQ }

	bufN := make([]float64, N)
	bufM := make([]float64, M)

	var iter int
	var residual float64
	var converged bool

	for iter = 0; iter < cfg.MaxIterations; iter++ {
		copy(logQPrev, logQ)

		if K > 1 {
			var wg sync.WaitGroup
			for k := 0; k < K; k++ {
				if weights[k] == 0.0 { continue }
				wg.Add(1)
				go func(kIdx int) {
					defer wg.Done()
					buf := make([]float64, N)
					for i := 0; i < M; i++ {
						for j := 0; j < N; j++ {
							buf[j] = (g[kIdx][j] - costFn(i, j)) * invEps
						}
						f[kIdx][i] = eps*logQ[i] - eps*LogSumExp(buf)
					}
				}(k)
			}
			wg.Wait()

			for k := 0; k < K; k++ {
				if weights[k] == 0.0 { continue }
				wg.Add(1)
				go func(kIdx int) {
					defer wg.Done()
					buf := make([]float64, M)
					for j := 0; j < N; j++ {
						for i := 0; i < M; i++ {
							buf[i] = (f[kIdx][i] - costFn(i, j)) * invEps
						}
						g[kIdx][j] = eps*logP[kIdx][j] - eps*LogSumExp(buf)
					}
				}(k)
			}
			wg.Wait()
		} else {
			for k := 0; k < K; k++ {
				if weights[k] == 0.0 { continue }
				for i := 0; i < M; i++ {
					for j := 0; j < N; j++ {
						bufN[j] = (g[k][j] - costFn(i, j)) * invEps
					}
					f[k][i] = eps*logQ[i] - eps*LogSumExp(bufN)
				}
			}

			for k := 0; k < K; k++ {
				if weights[k] == 0.0 { continue }
				for j := 0; j < N; j++ {
					for i := 0; i < M; i++ {
						bufM[i] = (f[k][i] - costFn(i, j)) * invEps
					}
					g[k][j] = eps*logP[k][j] - eps*LogSumExp(bufM)
				}
			}
		}

		for i := 0; i < M; i++ {
			logQ[i] = 0.0
			for k := 0; k < K; k++ {
				if weights[k] == 0.0 { continue }
				for j := 0; j < N; j++ {
					bufN[j] = (g[k][j] - costFn(i, j)) * invEps
				}
				logQ[i] += weights[k] * LogSumExp(bufN)
			}
		}

		lseTotal := LogSumExp(logQ)
		for i := 0; i < M; i++ { logQ[i] -= lseTotal }

		residual = 0.0
		for i := 0; i < M; i++ {
			residual += math.Abs(math.Exp(logQ[i]) - math.Exp(logQPrev[i]))
		}

		if residual < cfg.Tolerance {
			converged = true
			iter++
			break
		}
	}

	q := make([]float64, M)
	for i := 0; i < M; i++ { q[i] = math.Exp(logQ[i]) }

	subcosts := make([]float64, K)
	weightedCost := 0.0
	for k := 0; k < K; k++ {
		cost_k := 0.0
		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				c_ij := costFn(i, j)
				logP_ij := (f[k][i] + g[k][j] - c_ij) * invEps
				cost_k += math.Exp(logP_ij) * c_ij
			}
		}
		subcosts[k] = cost_k
		weightedCost += weights[k] * cost_k
	}

	return &BarycenterResult{
		Barycenter:    q,
		LogBarycenter: logQ,
		Iterations:    iter,
		FinalResidual: residual,
		Converged:     converged,
		Subcosts:      subcosts,
		WeightedCost:  weightedCost,
	}, nil
}

// ============================================================================
// 7. STOCHASTIC MINI-BATCH SINKHORN (STREAMING GENEVAY FRAMEWORK)
// ============================================================================

type Sampler func(batchSize int) [][]float64

type StreamingConfig struct {
	Epsilon      float64
	BatchSize    int
	Steps        int
	NumFeatures  int
	LearningRate float64
}

// StreamingContinuousSinkhorn menyelesaikan Optimal Transport skala jutaan / aliran kontinu
func StreamingContinuousSinkhorn(
	sampleX, sampleY Sampler,
	dim int,
	cfg StreamingConfig,
) (float64, error) {
	if dim <= 0 || cfg.BatchSize <= 0 || cfg.Steps <= 0 {
		return 0, errors.New("parameter konfigurasi tidak valid")
	}
	L := cfg.NumFeatures
	if L <= 0 { L = 256 }
	B := cfg.BatchSize
	eps := cfg.Epsilon
	invEps := 1.0 / eps

	W := make([][]float64, L)
	b := make([]float64, L)
	for l := 0; l < L; l++ {
		W[l] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			W[l][d] = rand.NormFloat64() * 0.3
		}
		b[l] = rand.Float64() * 2.0 * math.Pi
	}

	theta := make([]float64, L)
	mTheta := make([]float64, L)
	vTheta := make([]float64, L)

	phiX := make([][]float64, B)
	for i := range phiX { phiX[i] = make([]float64, L) }

	scaleRFF := math.Sqrt(2.0 / float64(L))
	bufB := make([]float64, B)
	wAccum := make([]float64, B)

	const beta1, beta2, epsAdam = 0.9, 0.999, 1e-8

	evalU := func(x []float64) float64 {
		val := 0.0
		for l := 0; l < L; l++ {
			dot := 0.0
			for d := 0; d < dim; d++ { dot += W[l][d] * x[d] }
			val += theta[l] * scaleRFF * math.Cos(dot + b[l])
		}
		return val
	}

	for step := 1; step <= cfg.Steps; step++ {
		batchX := sampleX(B)
		batchY := sampleY(B)

		uVals := make([]float64, B)
		for i := 0; i < B; i++ {
			uVal := 0.0
			for l := 0; l < L; l++ {
				dot := 0.0
				for d := 0; d < dim; d++ { dot += W[l][d] * batchX[i][d] }
				phi := scaleRFF * math.Cos(dot + b[l])
				phiX[i][l] = phi
				uVal += theta[l] * phi
			}
			uVals[i] = uVal
		}

		gradTheta := make([]float64, L)

		for l := 0; l < L; l++ {
			meanPhi := 0.0
			for i := 0; i < B; i++ { meanPhi += phiX[i][l] }
			gradTheta[l] = meanPhi / float64(B)
		}

		for i := 0; i < B; i++ { wAccum[i] = 0.0 }

		for j := 0; j < B; j++ {
			for i := 0; i < B; i++ {
				distSq := 0.0
				for d := 0; d < dim; d++ {
					diff := batchX[i][d] - batchY[j][d]
					distSq += diff * diff
				}
				bufB[i] = (uVals[i] - distSq) * invEps
			}
			lse := LogSumExp(bufB)
			for i := 0; i < B; i++ {
				wAccum[i] += math.Exp(bufB[i] - lse)
			}
		}

		invB := 1.0 / float64(B)
		for i := 0; i < B; i++ {
			factor := wAccum[i] * invB
			for l := 0; l < L; l++ {
				gradTheta[l] -= factor * phiX[i][l]
			}
		}

		for l := 0; l < L; l++ {
			g := gradTheta[l]
			mTheta[l] = beta1*mTheta[l] + (1.0-beta1)*g
			vTheta[l] = beta2*vTheta[l] + (1.0-beta2)*g*g
			mHat := mTheta[l] / (1.0 - math.Pow(beta1, float64(step)))
			vHat := vTheta[l] / (1.0 - math.Pow(beta2, float64(step)))

			theta[l] += cfg.LearningRate * mHat / (math.Sqrt(vHat) + epsAdam)
		}
	}

	const B_eval = 500
	evalX := sampleX(B_eval)
	evalY := sampleY(B_eval)

	uEval := make([]float64, B_eval)
	for i := 0; i < B_eval; i++ { uEval[i] = evalU(evalX[i]) }

	vEval := make([]float64, B_eval)
	bufEval := make([]float64, B_eval)
	for j := 0; j < B_eval; j++ {
		for i := 0; i < B_eval; i++ {
			distSq := 0.0
			for d := 0; d < dim; d++ {
				diff := evalX[i][d] - evalY[j][d]
				distSq += diff * diff
			}
			bufEval[i] = (uEval[i] - distSq) * invEps
		}
		vEval[j] = -eps * (LogSumExp(bufEval) - math.Log(float64(B_eval)))
	}

	meanU := 0.0
	for _, u_i := range uEval { meanU += u_i }
	meanU /= float64(B_eval)

	meanV := 0.0
	for _, v_j := range vEval { meanV += v_j }
	meanV /= float64(B_eval)

	return meanU + meanV, nil
}