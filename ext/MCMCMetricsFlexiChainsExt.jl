module MCMCMetricsFlexiChainsExt

import MCMCMetrics, FlexiChains

for name in MCMCMetrics._PARAMETER_DIAGNOSTICS
    @eval function MCMCMetrics.$name(chain::FlexiChains.FlexiChain;
        parameters=FlexiChains.parameters(chain), kwargs...)
        MCMCMetrics._parameterwise(MCMCMetrics.$name, parameters; kwargs...) do key
            parent(getindex(chain, FlexiChains.Parameter(key); stack=true))
        end
    end
end

end
