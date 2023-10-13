User-defined Migration Rules
============================

|tool_name| uses migration rules to migrate CUDA\* code to SYCL\* code.
There are three categories of migration rules used by the tool:

* **Default migration rules.** A set of built-in migration rules used by the
  |tool_name| for all migrations.
* **Optional predefined migration rules.**  A set of predefined migration rules
  that can optionally be used for migration. Available predefined migration rules
  are in the *extensions/opt_rules* folder on the installation path of the
  |tool_name|.
* **User-defined migration rules.** Custom migration rules defined by the user.
  User-defined migration rules extend the migration capability of the
  |tool_name| and can target the migration of specific CUDA syntax to
  SYCL syntax.

Specify Migration Rule Files
----------------------------

To specify a predefined or user-defined migration rule file for use in migration,
use the ``–rule-file`` command line option with your migration command.

The ``–rule-file`` option can be used multiple times with a single command to
specify multiple migration rule files. For example:

.. code-block:: none

   dpct sample.cu --rule-file=rule_file1.YAML --rule-file=rule_file2.YAML


See the :ref:`cmd_opt_ref` for additional information.

Write User-defined Migration Rules
----------------------------------

Migration rules are specified in YAML files. A single rule file may contain multiple migration rules. To define a rule, use the following
<key>: <value> pairs:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Key
     - Value
     - Description
   * - Rule
     - String value
     - Required. Specifies the unique name of the rule.
   * - Priority
     - ``Takeover`` | ``Default`` | ``Fallback``
     - Required. Specifies the priority of the rule: ``Takeover`` > ``Default`` > ``Fallback``.
       When there are rule conflicts, the rule with higher priority will take precedence.
   * - Kind
     - ``Macro`` | ``API`` | ``Header`` | ``Type`` | ``Class`` | ``Enum`` | ``DisableAPIMigration`` | ``PatternRewriter``
     - Required. Specifies the rule type.
   * - In
     - String value
     - Required. Specifies the target name in the input source code.
   * - Out
     - String value
     - Required. Specifies the final format in the output source code.
   * - Includes
     - List of header files
     - Required. Specifies the header files which should be included in the output source code. The value can be an empty list.
   * - EnumName
     - String value
     - Specifies the name of an enum for an Enum rule type.
   * - Fields
     - String value
     - Specifies the migration rule of fields in a Class rule type.
   * - Methods
     - String value
     - Specifies the migration rule of methods in a Class rule type.
   * - Prefix
     - String value
     - Specifies the prefix of a Header rule type. For example: ``#ifdef ...``
   * - Postfix
     - String value
     - Specifies the postfix of a Header rule type. For example: ``#endif ...``
   * - Subrules
     - String value
     - Specifies the subrules for the PatternRewriter rule type.

For example, the following user-defined migration rule file demonstrates different
rule types. The behavior of each rule is explained in the corresponding comment:

.. code-block:: none

   ---                                                    # [YAML syntax] Begin the document
   - Rule: rule_forceinline                               # Rule to migrate "__forceinline__" to "inline"
     Kind: Macro                                          # Rule type
     Priority: Takeover                                   # Rule priority
     In: __forceinline__                                  # Target macro name in the input source code
     Out: inline                                          # Migrated name of the macro in the output source code
     Includes: ["header1.h", "\"header2.h\""]             # List of header file names which the new macro depends on
   - Rule: rule_foo                                       # Rule to migrate "foo(a,b)" to "int *new_ptr=bar(*b)"
     Kind: API
     Priority: Takeover
     In: foo                                              # Target function name in the input source code
     Out: $type_name_of($2) *new_ptr = bar($deref($1))    # Format of the migrated result in the output source code
     Includes: ["<header3>"]
   - Rule: rule_cmath                                     # Rule to migrate "include<cmath>" to "#include<mymath>"
     Kind: Header
     Priority: Takeover
     In: cmath
     Out: mymath
     Prefix: "#ifdef USE_MYMATH\n"                        # Add prefix before "#include<mymath>"
     Postfix: "#endif\n"                                  # Add postfix after "#include<mymath>"
     Includes: [""]
   - Rule: rule_classA                                    # Rule to migrate "classA" to "classB"
     Kind: Class
     Priority: Takeover
     In: classA
     Out: classB
     Includes: []
     Fields:                                              # Specify the migration rule of fields of classA
       - In: fieldA                                       # Migrate classA.fieldA to getter and setter
         OutGetter: get_a                                 # Migrate value reference of classA.fieldA to classB.get_a()
         OutSetter: set_a                                 # Migrate value assignment of classA.fieldA to classB.set_a()
       - In: fieldC
         Out: fieldD                                      # Migrate classA.fieldC to classB.fieldD
     Methods:
       - In: methodA
         Out: a.methodB($2)                               # Migrate classA.methodA(x,y) to a.methodB(y)
   - Rule: rule_Fruit                                     # Rule to migrate "Fruit:apple" to "Fruit:pineapple"
     Kind: Enum
     Priority: Takeover
     EnumName: Fruit
     In: apple
     Out: pineapple
     Includes: ["fruit.h"]
   - Rule: type_rule                                      # Migrate "OldType" to "NewType"
     Kind: Type
     Priority: Takeover
     In: OldType
     Out: NewType
     Includes: []
   - Rule: disable_rule                                   # Disable the migration of an API
     Kind: DisableAPIMigration
     Priority: Takeover
     In: foo                                              # Disable the migration of foo
     Out: ""
     Includes: []
   - Rule: post_migration_rewriter_rule                   # Post-migration pattern rewrite rule which uses nested string pattern search and replace to find and update strings in the migrated code
     Kind: PatternRewriter
     Priority: Takeover
     In: my_max(${args});                                 # Match pattern "my_max(...);" and save the arbitrary string between "my_max(" and ");" as ${args}
                                                          # "args" can be a user-defined name which will be referenced by "Out" and "Subrules".
     Out: my_min(${args});                                # Replace the pattern string to "my_min(${args});"
     Includes: []
     Subrules:
       args:                                              # Specify the subrule to apply to ${args}. Where args is the user-defined name which is defined in "In".
         In: a                                            # Match pattern "a" in ${args}
         Out: b                                           # Replace the pattern string to "b" in ${args}
   ...                                                    # [YAML syntax] End the document


Grammar for Out Key in a User-defined API Migration Rule
--------------------------------------------------------

To describe the value format for the ``Out`` key in a migration rule of
``Kind: API``, use the following Backus-Naur form grammar:

.. code-block:: none

   OutValue::= Token | Token OutValue       # OutValue is the value for the “out” key
   Token::= AnyString | Keyword             # AnyString is a string provided by the user
   Keyword::= ArgIndex
      | $queue                              # Represents the queue string
      | $context                            # Represents the context string
      | $device                             # Represents the device string
      | $deref(ArgIndex)                    # The dereferenced value of the argument
      | $type_name_of(ArgIndex)             # The type name of the argument
      | $deref_type(ArgIndex)               # The dereferenced type name of the argument
      | $addr_of(ArgIndex)                  # The address of the argument
   ArgIndex::= $Int                         # Int should be a greater than zero integer


The following scenario describes how the tool makes use of a user-defined
migration rule that uses this grammar to migrate code.

Consider the following user-defined API migration rule:

.. code-block:: none

   - Rule: rule_foo
     Kind: API
     Priority: Takeover
     In: foo
     Out: $type_name_of($2) new_ptr = bar($deref($1), $3)
     Includes: [“<header3>”]

If the input source code contains a function call that matches the rule, the
tool parses the value of the ``In`` and ``Out`` keys and builds a keyword mapping
between the input and output source code. For example, with input source code:

.. code-block:: none

   int *ptr, *ptr2;
   foo(ptr, ptr2, 30);

The tool creates the following mapping:

.. list-table::
   :widths: 30 40 30
   :header-rows: 1

   * - Keyword
     - Input Source Code Match
     - Migration Result
   * - ``$1``
     - ``ptr``
     - ``ptr``
   * - ``$2``
     - ``ptr2``
     - ``ptr2``
   * - ``$3``
     - ``30``
     - ``30``
   * - ``$type_name_of($2)``
     - N/A
     - ``int*``
   * - ``$deref($1)``
     - N/A
     - ``*ptr``

Using this mapping, the tool migrates the input source code into the following
output source code:

.. code-block:: none

   int *ptr, *ptr2;
   int * new_ptr = bar(*ptr, 30);
