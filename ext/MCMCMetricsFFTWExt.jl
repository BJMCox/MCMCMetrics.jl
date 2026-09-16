module MCMCMetricsFFTWExt

import FFTW, MCMCMetrics

function MCMCMetrics._fft_autocov(z::AbstractMatrix{T}) where {T<:Union{Float32,Float64}}
    n, m = size(z)
    # Zero padding avoids circular correlations. FFTW's inverse includes 1/N.
    padded = zeros(T, nextpow(2, 2n - 1), m)
    padded[1:n, :] .= z
    transformed = FFTW.rfft(padded, 1)
    covariance = FFTW.irfft(abs2.(transformed), size(padded, 1), 1)
    vec(sum(view(covariance, 1:n, :); dims=2)) ./ T(n) ./ T(m)
end

end
