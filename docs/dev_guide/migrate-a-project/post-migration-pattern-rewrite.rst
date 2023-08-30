.. _pattern_rewriter_rule_ref:

Post-migration Pattern-Rewrite
============================

|tool_name| supports post-migration pattern-rewrite which can apply nested
string pattern search and replacement to the migrated code. The pattern-rewrite
feature is integrated in the user-defined rule feature :ref:`_user_define_rule_ref`.
The pattern-rewrite feature can be enabled by adding a rule with kind
"PatternRewriter" into the rule YAML file and enable the rule file with ``–rule-file``
command line option.

Example of a PatternRewriter Rule
--------------------------------------------------------
.. code-block:: none

  - Rule: rule_post
    Kind: PatternRewriter
    Priority: Takeover
    In: my_max(${args});               # Match pattern "my_max(...);" and save the arbitrary string between "my_max(" and ");" as ${args}
    Out: my_min(${args});              # Replace the pattern string to "my_min(${args});"
    Includes: []
    Subrules:
      args:                            # Specify the subrule to apply to ${args}
        In: a                          # Match pattern "a" in ${args}
        Out: b                         # Replace the pattern string to "b" in ${args}

After applying the rule above, a string "my_max(a, b);" in the migrated code
will be replaced to "my_min(b, b);" by the post-migration pattern rewriter of
|tool_name|.

