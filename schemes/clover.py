# CLOVER = CLOsed-loop VERifier-driven workflow optimization

from myopto.utils.llm_router import set_role_models
set_role_models(
    metaoptimizer="gpt-5-mini",
    optimizer="gpt-5-nano",
    executor="gpt-5-nano",
)

from myopto.utils.usage import configure_usage, reset_usage, get_total_cost, get_cost_by_role
configure_usage(True)
reset_usage()


from myopto.utils.llm_call import _llm_prep_function

llm = _llm_prep_function(role="executor")
out = llm("Hello")

print(get_total_cost())
print(get_cost_by_role())
