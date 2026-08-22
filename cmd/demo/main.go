package main

import (
	"fmt"
	"math/rand"
	
	"algoscale"
)

func main() {
	fmt.Println("Memulai Pengujian Terpadu algoscale...")

	// 1. Inversi Matriks
	A := algoscale.NewMatrix(2, 2)
	A.Set(0, 0, 4); A.Set(0, 1, 7)
	A.Set(1, 0, 2); A.Set(1, 1, 6)
	invA, _ := algoscale.InvertSmallMatrix(A)
	fmt.Printf("[1] Inversi Selesai: A^-1[0,0] = %.2f\n", invA.At(0, 0))

	// 2. Unbalanced Sinkhorn
	K := algoscale.NewMatrix(2, 2)
	K.Set(0, 0, 0.9); K.Set(0, 1, 0.3)
	K.Set(1, 0, 0.3); K.Set(1, 1, 0.9)
	resUnb, _ := algoscale.UnbalancedSinkhorn(K, []float64{0.5, 0.5}, []float64{0.4, 0.6}, 
		algoscale.UnbalancedConfig{Epsilon: 0.1, Tau1: 1.0, Tau2: 1.0, MaxIterations: 100, Tolerance: 1e-4})
	fmt.Printf("[2] Unbalanced Converged: %t\n", resUnb.Converged)

	// 3. Matrix-Free Log-Domain Sinkhorn
	costFn := func(i, j int) float64 { return float64((i - j) * (i - j)) * 0.1 }
	resLog, _ := algoscale.LogSinkhornStreaming(4, 4, 
		[]float64{-1.38, -1.38, -1.38, -1.38}, []float64{-1.38, -1.38, -1.38, -1.38}, 
		costFn, algoscale.LogSinkhornConfig{Epsilon: 0.1, Tau1: 1.0, Tau2: 1.0, MaxIterations: 100, Tolerance: 1e-4})
	fmt.Printf("[3] Log-Domain Matrix-Free Converged: %t\n", resLog.Converged)

	// 4. Debiased Sinkhorn Divergence
	pts1 := [][]float64{{0, 0}, {1, 1}}
	pts2 := [][]float64{{2, 2}, {3, 3}}
	euclid := func(a, b []float64) float64 { return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) }
	div, _ := algoscale.SinkhornDivergence(pts1, pts2, euclid, algoscale.LogSinkhornConfig{Epsilon: 0.2, MaxIterations: 50, Tolerance: 1e-4})
	fmt.Printf("[4] Debiased Metric Distance: %f\n", div)

	// 5. Wasserstein Barycenters
	resBar, _ := algoscale.ComputeBarycenterLogDomain([][]float64{{1, 0, 0}, {0, 0, 1}}, []float64{0.5, 0.5}, 
		costFn, 3, 3, algoscale.BarycenterConfig{Epsilon: 1.0, MaxIterations: 100, Tolerance: 1e-4})
	fmt.Printf("[5] Barycenter Mean Vector: %v\n", resBar.Barycenter)

	// 6. Stochastic Streaming Sinkhorn (Jutaan Titik)
	sX := func(n int) [][]float64 {
		res := make([][]float64, n)
		for i := range res { res[i] = []float64{rand.NormFloat64()} }
		return res
	}
	sY := func(n int) [][]float64 {
		res := make([][]float64, n)
		for i := range res { res[i] = []float64{rand.NormFloat64() + 2.0} }
		return res
	}
	costStoch, _ := algoscale.StreamingContinuousSinkhorn(sX, sY, 1, 
		algoscale.StreamingConfig{Epsilon: 0.2, BatchSize: 64, Steps: 100, NumFeatures: 64, LearningRate: 0.05})
	fmt.Printf("[6] Stochastic Streaming Cost: %f\n", costStoch)
}