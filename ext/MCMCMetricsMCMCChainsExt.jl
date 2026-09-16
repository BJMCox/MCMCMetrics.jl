module MCMCMetricsMCMCChainsExt

import MCMCMetrics, MCMCChains

for name in MCMCMetrics._PARAMETER_DIAGNOSTICS
    @eval function MCMCMetrics.$name(chain::MCMCChains.Chains;
        parameters=MCMCChains.names(chain, :parameters), kwargs...)
        indices = Dict(key => i for (i, key) in enumerate(MCMCChains.names(chain)))
        MCMCMetrics._parameterwise(MCMCMetrics.$name, parameters; kwargs...) do key
            view(parent(chain.value), :, indices[key], :)
        end
    end
end

end
