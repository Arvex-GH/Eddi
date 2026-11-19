"""Built-in curriculum used to train Eddi from zero.

The curriculum is a small, self-contained knowledge corpus (facts + prose) split
into topic packs.  It is deliberately *not* the evaluation set: ``eddi.eval``
holds out separate sentences so the reported perplexity/accuracy are honest.

Packs
-----
``core``        identity, computing basics, memory hierarchy
``science``     physics, astronomy, biology, chemistry, mathematics
``computing``   programming languages, C++ / C# / Python, systems, data
``reasoning``   logic, method, statistics, problem solving
``geography``   continents, oceans, planets, units of measure
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

Fact = Tuple[str, str, str]

# --------------------------------------------------------------------- facts
CORE_FACTS: List[Fact] = [
    ("Eddi", "name", "Eddi"),
    ("Eddi", "type", "local_hyperdimensional_agent"),
    ("Eddi", "runs_on", "consumer_hardware_no_gpu"),
    ("Eddi", "architecture", "HDC_plus_disk_backed_context"),
    ("Eddi", "uses", "hyperdimensional_computing"),
    ("Eddi", "memory_model", "writable_hypervectors_no_backprop"),
    ("Eddi", "context_capacity", "one_million_tokens_mmap"),
    ("HDC", "core_mechanism", "hyperdimensional_vector_binding"),
    ("HyperVector", "dimension", "1024_bits"),
    ("HDC_bind", "algebra", "XOR"),
    ("HDC_bundle", "algebra", "majority_vote"),
    ("HDC_cleanup", "purpose", "snap_noisy_vector_to_nearest_symbol"),
    ("Transformer", "core_mechanism", "self_attention"),
    ("Transformer", "trained_by", "backpropagation"),
    ("CPU", "role", "executes_instructions"),
    ("GPU", "role", "parallel_matrix_compute"),
    ("VRAM", "role", "high_bandwidth_gpu_memory"),
    ("RAM", "role", "volatile_working_memory"),
    ("SSD", "role", "persistent_storage"),
    ("register", "speed_rank", "fastest_memory"),
    ("RAM", "speed_rank", "slower_than_cache"),
    ("SSD", "speed_rank", "slower_than_RAM"),
    ("operating_system", "role", "manage_hardware_and_services"),
    ("compiler", "role", "translate_source_to_machine_code"),
    ("interpreter", "role", "execute_source_directly"),
    ("algorithm", "definition", "finite_steps_solving_a_problem"),
    ("complexity", "measure", "growth_of_runtime_with_input"),
]

SCIENCE_FACTS: List[Fact] = [
    ("Earth", "orbits", "Sun"),
    ("Moon", "orbits", "Earth"),
    ("Mars", "orbits", "Sun"),
    ("Jupiter", "orbits", "Sun"),
    ("Earth", "type", "planet"),
    ("Mars", "type", "planet"),
    ("Jupiter", "type", "planet"),
    ("Sun", "type", "star"),
    ("Moon", "type", "natural_satellite"),
    ("Water", "chemical_formula", "H2O"),
    ("Carbon_dioxide", "chemical_formula", "CO2"),
    ("Salt", "chemical_formula", "NaCl"),
    ("speed_of_light", "approx_value", "299792458_m_per_s"),
    ("gravity", "discovered_by", "Newton"),
    ("Earth", "gravity", "9_point_81_m_per_s2"),
    ("earthquake", "measured_by", "Richter_scale"),
    ("atom", "nucleus_contains", "protons_and_neutrons"),
    ("photosynthesis", "input", "light_water_and_CO2"),
    ("photosynthesis", "output", "glucose_and_oxygen"),
    ("DNA", "shape", "double_helix"),
    ("human_body", "blood_pump", "heart"),
    ("oxygen", "carried_by", "hemoglobin"),
    ("cell", "energy_currency", "ATP"),
]

COMPUTING_FACTS: List[Fact] = [
    ("Python", "type", "interpreted_programming_language"),
    ("C++", "type", "compiled_systems_language"),
    ("C#", "type", "managed_language_for_dotnet"),
    ("Rust", "type", "memory_safe_systems_language"),
    ("Python", "created_by", "Guido_van_Rossum"),
    ("C++", "created_by", "Bjarne_Stroustrup"),
    ("C#", "created_by", "Anders_Hejlsberg"),
    ("Python", "use_case", "automation_and_data_science"),
    ("C++", "use_case", "games_and_systems"),
    ("C#", "use_case", "windows_and_unity_apps"),
    ("C++_RAII", "meaning", "Resource_Acquisition_Is_Initialization"),
    ("C++_unique_ptr", "ownership", "exclusive_single_owner"),
    ("C++_shared_ptr", "ownership", "reference_counted_shared"),
    ("C++_weak_ptr", "purpose", "non_owning_observer_breaking_cycles"),
    ("C++_std_move", "purpose", "cast_to_rvalue_reference"),
    ("C++_constexpr", "meaning", "compile_time_constant_expression"),
    ("C++_concepts", "purpose", "constrain_template_parameters"),
    ("C++_std_span", "purpose", "non_owning_contiguous_view"),
    ("C++_std_jthread", "purpose", "joining_thread_with_stop_token"),
    ("C++_std_atomic", "purpose", "lock_free_thread_safe_operations"),
    ("C++_co_await", "feature", "coroutine_suspension_point"),
    ("C++_virtual", "purpose", "runtime_polymorphism_via_vtable"),
    ("C#_record", "semantics", "immutable_value_equality"),
    ("C#_struct", "allocation", "value_type_inline"),
    ("C#_Span_T", "purpose", "zero_allocation_memory_slice"),
    ("C#_LINQ", "purpose", "declarative_query_over_IEnumerable"),
    ("C#_async_await", "purpose", "non_blocking_state_machine"),
    ("C#_ValueTask_T", "purpose", "allocation_free_async_result"),
    ("C#_IAsyncEnumerable_T", "purpose", "asynchronous_streaming"),
    ("C#_CancellationToken", "purpose", "cooperative_cancellation"),
    ("C#_pattern_matching", "feature", "switch_expressions"),
    ("C#_stackalloc", "purpose", "allocate_on_stack"),
    ("C#_GC", "generations", "Gen0_Gen1_Gen2_LOH"),
    ("mmap", "purpose", "map_file_into_memory"),
    ("LSH", "purpose", "approximate_nearest_neighbour_bucketing"),
    ("hash_table", "average_lookup", "O_of_1"),
    ("binary_search", "runtime", "O_of_log_n"),
    ("merge_sort", "runtime", "O_of_n_log_n"),
    ("SQL", "role", "query_relational_databases"),
    ("JSON", "role", "text_data_interchange"),
    ("HTTP", "port_default", "80"),
    ("HTTPS", "port_default", "443"),
]

REASONING_FACTS: List[Fact] = [
    ("logic", "core_rule", "modus_ponens"),
    ("modus_ponens", "form", "if_P_implies_Q_and_P_then_Q"),
    ("induction", "direction", "specific_to_general"),
    ("deduction", "direction", "general_to_specific"),
    ("abduction", "direction", "observation_to_best_explanation"),
    ("Occams_razor", "principle", "prefer_simplest_explanation"),
    ("mean", "definition", "sum_divided_by_count"),
    ("median", "definition", "middle_value_of_sorted_data"),
    ("mode", "definition", "most_frequent_value"),
    ("correlation", "warning", "does_not_imply_causation"),
    ("bayes_theorem", "purpose", "update_belief_with_evidence"),
    ("overfitting", "symptom", "good_on_training_bad_on_test"),
    ("entropy", "measure", "uncertainty_of_a_distribution"),
    ("perplexity", "definition", "exponential_of_cross_entropy"),
]

GEOGRAPHY_FACTS: List[Fact] = [
    ("Berlin", "capital_of", "Germany"),
    ("Paris", "capital_of", "France"),
    ("Tokyo", "capital_of", "Japan"),
    ("Germany", "continent", "Europe"),
    ("France", "continent", "Europe"),
    ("Japan", "continent", "Asia"),
    ("Nile", "longest_river_in", "Africa"),
    ("Everest", "highest_mountain_on", "Earth"),
    ("kilometer", "equals", "1000_meters"),
    ("hour", "equals", "60_minutes"),
    ("minute", "equals", "60_seconds"),
    ("day", "equals", "24_hours"),
]

# ---------------------------------------------------------------------- prose
CORE_PROSE: List[str] = [
    "Eddi is a local hyperdimensional agent that runs on your own computer without a GPU.",
    "Eddi writes facts directly into memory as hypervectors, so learning needs no backpropagation.",
    "A writable knowledge base stores each fact as a subject, a relation and an object.",
    "Hyperdimensional computing binds concepts with XOR and bundles them with a majority vote.",
    "Binding is self inverse, so unbinding a stored fact recovers the object vector exactly.",
    "The cleanup memory snaps a noisy recovered vector to the nearest known symbol.",
    "Eddi keeps a sliding window of recent tokens in RAM and streams older tokens to a memory mapped file.",
    "The disk backed context can hold one million tokens without growing the RAM footprint.",
    "A locality sensitive hash bucket finds a similar token in constant time on the mmap file.",
    "Every answer Eddi gives states whether it came from memory, from reasoning or from the language model.",
    "Eddi can be taught new facts in one step because memory is written, not trained.",
    "The language model is an interpolated n-gram model that mixes exact tokens with word type classes.",
    "A word type class groups runs, running and ran under the same stem so unseen contexts still match.",
    "Entropy sampling with a repetition penalty lets Eddi write whole sentences instead of loops.",
    "Perplexity measures how surprised a language model is by held out text, and lower is better.",
    "An agent should say I do not know rather than invent an answer.",
    "Good software is simple to read, easy to test and cheap to change.",
    "A cache trades capacity for speed, and the memory hierarchy goes registers, cache, RAM and SSD.",
    "The compiler translates source code into machine code before the program runs.",
    "An interpreter executes source code directly, one statement at a time.",
    "An algorithm is a finite list of steps that solves a problem for every valid input.",
    "Complexity describes how runtime grows when the input gets larger.",
    "A hash table offers constant time average lookup for keys.",
    "Binary search halves the remaining range on every step and runs in logarithmic time.",
    "Tests are executable documentation that protect behaviour during refactoring.",
    "Version control records the history of a project so changes can be reviewed and undone.",
    "Continuous integration builds and tests every change automatically.",
]

SCIENCE_PROSE: List[str] = [
    "The Earth orbits the Sun once every year at an average speed of thirty kilometers per second.",
    "The Moon orbits the Earth in about twenty seven days and causes the ocean tides.",
    "Gravity attracts masses toward each other, and Earth pulls falling objects down at 9.81 meters per second squared.",
    "Water freezes at zero degrees Celsius and boils at one hundred degrees Celsius at sea level.",
    "Photosynthesis turns light, water and carbon dioxide into glucose and oxygen.",
    "DNA stores the instructions of life in a double helix of four bases.",
    "A cell uses ATP as its energy currency for chemical work.",
    "The speed of light in vacuum is about 299792458 meters per second.",
    "Sound travels much slower than light because it needs a medium to carry vibrations.",
    "An atom has a nucleus of protons and neutrons surrounded by electrons.",
    "The Sun is a star that fuses hydrogen into helium and releases light and heat.",
    "Energy is never created or destroyed, it only changes form.",
    "The human heart pumps blood through the body and carries oxygen to every tissue.",
    "Mars is called the red planet because iron oxide covers its surface.",
    "Jupiter is the largest planet of the solar system and has many moons.",
    "The Richter scale measures the strength of an earthquake.",
]

COMPUTING_PROSE: List[str] = [
    "C++ gives the programmer control over memory layout and object lifetime.",
    "RAII ties resource lifetime to object scope so destructors release memory and files automatically.",
    "A unique pointer owns one object exclusively, while a shared pointer counts its owners.",
    "A weak pointer observes a shared pointer without keeping the object alive and breaks cycles.",
    "Move semantics transfer resources instead of copying them.",
    "Constexpr evaluates expressions at compile time, and consteval allows only immediate evaluation.",
    "Concepts constrain template parameters and produce readable compiler errors.",
    "A span is a non owning view over a contiguous range of memory.",
    "Coroutines suspend and resume execution without blocking a thread.",
    "Atomics provide lock free operations for concurrent code.",
    "C# records give immutable data types with value equality and concise syntax.",
    "A struct in C# is a value type that is usually allocated inline.",
    "Span of T is a stack only slice that avoids allocations.",
    "LINQ builds declarative queries over any enumerable sequence.",
    "Async and await turn asynchronous code into readable linear code with a state machine.",
    "A value task avoids allocation when an asynchronous result is usually synchronous.",
    "An async enumerable streams values asynchronously with await foreach.",
    "A cancellation token cooperates with cancellation without killing threads.",
    "Python is interpreted, dynamic and famous for readability and a huge standard library.",
    "Rust enforces memory safety at compile time with ownership and borrowing.",
    "A relational database stores rows in tables and SQL queries them declaratively.",
    "JSON is a compact text format for exchanging structured data between programs.",
    "HTTP is the request and response protocol of the web and HTTPS encrypts it with TLS.",
    "A memory mapped file lets a program read and write a file as if it were an array.",
    "Locality sensitive hashing places similar items in the same bucket with high probability.",
]

REASONING_PROSE: List[str] = [
    "Modus ponens says that if P implies Q and P is true, then Q must be true.",
    "Deduction moves from general rules to specific conclusions.",
    "Induction moves from specific observations to a general rule and is never certain.",
    "Abduction chooses the most plausible explanation for an observation.",
    "Occam's razor prefers the simplest explanation that fits the evidence.",
    "The mean adds all values and divides by the count.",
    "The median is the middle value of sorted data and resists outliers.",
    "Correlation does not imply causation, so a third factor may drive both variables.",
    "Bayes theorem updates a prior belief with new evidence.",
    "Overfitting means a model memorises the training data and fails on new data.",
    "Entropy measures the uncertainty of a probability distribution.",
    "Perplexity is the exponential of cross entropy and lower values mean better prediction.",
    "A good experiment changes one variable at a time and keeps the rest fixed.",
    "A claim without evidence can be rejected without evidence.",
    "When two explanations predict the same thing, prefer the one with fewer assumptions.",
]

GEOGRAPHY_PROSE: List[str] = [
    "Berlin is the capital of Germany and lies in Europe.",
    "Paris is the capital of France and sits on the river Seine.",
    "Tokyo is the capital of Japan and is one of the largest cities in the world.",
    "The Nile flows through Africa and is one of the longest rivers on Earth.",
    "Mount Everest is the highest mountain on Earth at 8849 meters.",
    "One kilometer equals one thousand meters, and one hour equals sixty minutes.",
    "A day has twenty four hours, and a minute has sixty seconds.",
]

# The teaching curriculum lives in eddi/curriculum_packs/, one module per
# subject, so this file stays a registry rather than growing without bound.
from .curriculum_packs import mathematics as _mathematics

PACKS: Dict[str, Dict[str, List]] = {
    "core": {"facts": CORE_FACTS, "prose": CORE_PROSE},
    "mathematics": {"facts": _mathematics.FACTS, "prose": _mathematics.PROSE},
    "science": {"facts": SCIENCE_FACTS, "prose": SCIENCE_PROSE},
    "computing": {"facts": COMPUTING_FACTS, "prose": COMPUTING_PROSE},
    "reasoning": {"facts": REASONING_FACTS, "prose": REASONING_PROSE},
    "geography": {"facts": GEOGRAPHY_FACTS, "prose": GEOGRAPHY_PROSE},
}

ALL_PACKS = tuple(PACKS)


def pack_facts(pack: str) -> List[Fact]:
    return list(PACKS[pack]["facts"])


def pack_prose(pack: str) -> List[str]:
    return list(PACKS[pack]["prose"])


def all_facts(packs: Iterable[str] = ALL_PACKS) -> List[Fact]:
    out: List[Fact] = []
    for pack in packs:
        out.extend(pack_facts(pack))
    return out


def all_prose(packs: Iterable[str] = ALL_PACKS) -> List[str]:
    out: List[str] = []
    for pack in packs:
        out.extend(pack_prose(pack))
    return out
