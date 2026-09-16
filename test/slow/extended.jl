using MCMCMetrics, Random, Statistics, Test

function extended_process_checks(; repetitions=1024, n=4096)
    rng = Xoshiro(160926)
    results = []
    @testset "Batch precision process checks" begin
        for rho in (-0.5,0.0,0.8)
            finite_tau = 1 + 2sum((1-lag/n)*rho^lag for lag in 1:n-1)
            target = finite_tau/n
            averages = Float64[]
            errors = zeros(repetitions,2)
            for rep in 1:repetitions
                x = randn(rng,n)
                for i in 2:n
                    x[i] = rho*x[i-1] + sqrt(1-rho^2)*x[i]
                end
                push!(averages,mean(x))
                for (j,estimator) in enumerate((:batchmeans,:overlapping))
                    errors[rep,j] = mcse(x;estimator,batch_size=64,split=false)
                end
            end
            @test var(averages) ≈ target rtol=4sqrt(2/(repetitions-1))
            for (j,estimator) in enumerate((:batchmeans,:overlapping))
                ratio = mean(abs2,errors[:,j])/target
                coverage = count(abs.(averages) .<= 1.96.*errors[:,j])/repetitions
                # Width 64 has finite-batch bias, especially for positive rho.
                @test abs(ratio-1) < 0.15
                @test coverage >= 0.95-4sqrt(0.95*0.05/repetitions)
                push!(results,(family=:mean,rho,estimator,variance_ratio=ratio,coverage))
            end
        end
    end
    @testset "Importance ratio process checks" begin
        beta = 0.5
        averages, errors = Float64[], Float64[]
        target = exp(beta^2)*(1+beta^2)/n
        for _ in 1:repetitions
            x = randn(rng,n)
            w = exp.(beta.*x)
            push!(averages,sum(w.*x)/sum(w))
            push!(errors,importance_mcse(x,beta.*x;sampling=:iid))
        end
        ratio = mean(abs2,errors)/target
        coverage = count(abs.(averages.-beta) .<= 1.96.*errors)/repetitions
        @test abs(ratio-1) < 0.15
        @test var(averages) ≈ target rtol=4sqrt(2/(repetitions-1))
        @test coverage >= 0.95-4sqrt(0.95*0.05/repetitions)
        push!(results,(family=:importance,beta,variance_ratio=ratio,coverage))
    end
    return results
end

extended_process_checks()
