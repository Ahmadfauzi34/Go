package algoscale

import (
	"math"
	"testing"
)

func FuzzInvertSmallMatrix(f *testing.F) {
	// Seed initial corpus
	f.Add(2, []byte{1, 2, 3, 4})
	f.Add(3, []byte{1, 0, 0, 0, 1, 0, 0, 0, 1})

	f.Fuzz(func(t *testing.T, n int, data []byte) {
		if n <= 0 || n > 20 {
			return
		}
		if len(data) < n*n {
			return
		}

		m := NewMatrix(n, n)
		for i := 0; i < n*n; i++ {
			// convert byte to float64 including edge cases
			val := float64(data[i])
			if i%5 == 0 {
				val = math.NaN()
			} else if i%7 == 0 {
				val = math.Inf(1)
			} else if i%11 == 0 {
				val = math.MaxFloat64
			} else if i%13 == 0 {
				val = math.SmallestNonzeroFloat64
			}
			m.Data[i] = val
		}

		inv, err := InvertSmallMatrix(m)
		if err == nil && inv == nil {
			t.Errorf("InvertSmallMatrix returned nil result without error")
		}
	})
}

func FuzzLogSumExp(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			res := LogSumExp(nil)
			if !math.IsInf(res, -1) {
				t.Errorf("expected -Inf for empty slice, got %f", res)
			}
			return
		}

		vals := make([]float64, len(data))
		for i, b := range data {
			val := float64(b) - 128.0
			if i%3 == 0 {
				val = math.NaN()
			} else if i%5 == 0 {
				val = math.Inf(1)
			} else if i%7 == 0 {
				val = math.Inf(-1)
			}
			vals[i] = val
		}

		res := LogSumExp(vals)
		_ = res
	})
}

func FuzzLogSinkhornStreaming(f *testing.F) {
	f.Add(2, 2, 0.1, 1.0, 1.0, 10, 1e-3)
	f.Fuzz(func(t *testing.T, m, n int, eps, tau1, tau2 float64, maxIter int, tol float64) {
		if m <= 0 || m > 50 || n <= 0 || n > 50 {
			return
		}
		if math.IsNaN(eps) || math.IsNaN(tau1) || math.IsNaN(tau2) || math.IsNaN(tol) {
			return
		}

		logR := make([]float64, m)
		logC := make([]float64, n)
		for i := range logR {
			logR[i] = -math.Log(float64(m))
		}
		for j := range logC {
			logC[j] = -math.Log(float64(n))
		}

		costFn := func(i, j int) float64 {
			return float64((i - j) * (i - j))
		}

		cfg := LogSinkhornConfig{
			Epsilon:       eps,
			Tau1:          tau1,
			Tau2:          tau2,
			MaxIterations: maxIter,
			Tolerance:     tol,
		}

		res, err := LogSinkhornStreaming(m, n, logR, logC, costFn, cfg)
		if err == nil {
			if res == nil {
				t.Errorf("LogSinkhornStreaming returned nil result without error")
			}
		}
	})
}

func FuzzUnbalancedSinkhorn(f *testing.F) {
	f.Add(2, 2, 0.1, 1.0, 1.0, 10, 1e-3)
	f.Fuzz(func(t *testing.T, m, n int, eps, tau1, tau2 float64, maxIter int, tol float64) {
		if m <= 0 || m > 20 || n <= 0 || n > 20 {
			return
		}
		if math.IsNaN(eps) || math.IsNaN(tau1) || math.IsNaN(tau2) || math.IsNaN(tol) {
			return
		}

		K := NewMatrix(m, n)
		for i := 0; i < m*n; i++ {
			K.Data[i] = 0.5
		}
		r := make([]float64, m)
		c := make([]float64, n)
		for i := range r {
			r[i] = 1.0 / float64(m)
		}
		for j := range c {
			c[j] = 1.0 / float64(n)
		}

		cfg := UnbalancedConfig{
			Epsilon:       eps,
			Tau1:          tau1,
			Tau2:          tau2,
			MaxIterations: maxIter,
			Tolerance:     tol,
		}

		res, err := UnbalancedSinkhorn(K, r, c, cfg)
		if err == nil {
			if res == nil {
				t.Errorf("UnbalancedSinkhorn returned nil result without error")
			}
		}
	})
}
