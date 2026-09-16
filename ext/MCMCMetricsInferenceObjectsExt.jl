module MCMCMetricsInferenceObjectsExt

import MCMCMetrics, InferenceObjects, DimensionalData

for name in MCMCMetrics._PARAMETER_DIAGNOSTICS
    @eval begin
        function MCMCMetrics.$name(data::InferenceObjects.Dataset;
            parameters=keys(data), kwargs...)
            MCMCMetrics._parameterwise(MCMCMetrics.$name, parameters; kwargs...) do key
                x = data[key]
                draw, chain = DimensionalData.dimnum(x, (:draw, :chain))
                components = Tuple(i for i in 1:ndims(x) if i != draw && i != chain)
                PermutedDimsArray(parent(x), (draw, chain, components...))
            end
        end

        MCMCMetrics.$name(data::InferenceObjects.InferenceData; kwargs...) =
            MCMCMetrics.$name(data.posterior; kwargs...)
    end
end

end
