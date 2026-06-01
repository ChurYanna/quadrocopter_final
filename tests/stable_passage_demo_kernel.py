from tests.test_three_snake_dynamic_gates import TestThreeSnakeDynamicGates


def run_stable_mixed_passage_demo(test_case):
    """Run the stable mixed-obstacle passage executor.

    Scenario tests should provide only configuration and scene assets, then
    call this kernel.  Keeping this as the single entry point makes new demos
    less likely to patch or fork the proven low-level control path.
    """

    return TestThreeSnakeDynamicGates.run_stable_passage_kernel(test_case)
