"""Comprehensive C++ (C++11..C++23) and C# (C# 8..12 / .NET 8) curriculum for Eddi."""

from typing import List, Tuple
from .agent import EddiAgent

CPP_CS_FACTS: List[Tuple[str, str, str]] = [
    # C++ Core & Memory
    ("C++_RAII", "meaning", "Resource_Acquisition_Is_Initialization"),
    ("C++_unique_ptr", "ownership", "exclusive_single_owner_smart_pointer"),
    ("C++_shared_ptr", "ownership", "reference_counted_shared_smart_pointer"),
    ("C++_weak_ptr", "purpose", "non_owning_observer_breaking_cycles"),
    ("C++_std_move", "purpose", "cast_to_rvalue_reference_for_move_semantics"),
    ("C++_constexpr", "evaluation", "compile_time_constant_expression"),
    ("C++_consteval", "evaluation", "immediate_compile_time_only_function"),
    ("C++_string_view", "purpose", "non_owning_zero_copy_string_slice"),
    ("C++_std_span", "purpose", "non_owning_contiguous_array_view"),
    ("C++_concepts", "purpose", "constrain_template_parameters_at_compile_time"),
    ("C++_std_jthread", "purpose", "joining_thread_with_stop_token_support"),
    ("C++_std_atomic", "purpose", "lock_free_thread_safe_operations"),
    ("C++_co_await", "feature", "C++20_coroutines_suspension_point"),
    ("C++_virtual", "purpose", "runtime_polymorphism_via_vtable"),
    ("C++_Rule_of_Zero", "principle", "use_RAII_types_so_no_custom_destructor_needed"),

    # C# Core & .NET
    ("C#_record", "semantics", "immutable_reference_type_with_value_equality"),
    ("C#_struct", "allocation", "value_type_stack_or_inline_allocated"),
    ("C#_Span_T", "purpose", "stack_only_zero_allocation_memory_slice"),
    ("C#_LINQ", "purpose", "declarative_query_operators_over_IEnumerable"),
    ("C#_async_await", "purpose", "non_blocking_asynchronous_state_machine"),
    ("C#_ValueTask_T", "purpose", "allocation_free_async_result_when_synchronous"),
    ("C#_IAsyncEnumerable_T", "purpose", "asynchronous_streaming_iteration_await_foreach"),
    ("C#_CancellationToken", "purpose", "cooperative_async_cancellation"),
    ("C#_pattern_matching", "feature", "switch_expressions_and_property_patterns"),
    ("C#_stackalloc", "purpose", "allocate_memory_block_on_stack"),
    ("C#_LibraryImport", "purpose", "source_generated_native_P_Invoke_interop"),
    ("C#_GC", "generations", "Gen0_Gen1_Gen2_and_LOH"),
]

CPP_CS_CORPUS = """
// C++20 Modern Patterns
#include <memory>
#include <vector>
#include <span>
#include <concepts>

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <Numeric T>
T sum_span(std::span<const T> data) {
    T total = 0;
    for (const auto& x : data) total += x;
    return total;
}

// C++ RAII & Smart Pointers
auto ptr = std::make_unique<int>(42);
auto shared = std::make_shared<std::vector<int>>(100, 0);

// C# 12 / .NET 8 Modern Patterns
public readonly record struct Point(double X, double Y);

public static async Task<int> FetchCountAsync(IAsyncEnumerable<int> stream, CancellationToken ct)
{
    int count = 0;
    await foreach (var item in stream.WithCancellation(ct))
    {
        count += item;
    }
    return count;
}

// C# Span<T> zero-allocation slicing
Span<int> slice = stackalloc int[16];
var positives = new[] { -1, 2, 3 }.Where(x => x > 0).Select(x => x * 2).ToArray();
"""


def teach_cpp_and_csharp(agent: EddiAgent) -> dict:
    for s, r, o in CPP_CS_FACTS:
        agent.teach(s, r, o)
    tokens = agent.ingest_text(CPP_CS_CORPUS)
    return {"facts": len(CPP_CS_FACTS), "tokens": tokens}
