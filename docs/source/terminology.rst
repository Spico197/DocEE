.. toctree::
   :maxdepth: 2
   :caption: Contents:


Terminology
===========


:Authors:
    Tong Zhu

:Last Update: Jan. 4th, 2021


Contents
--------

Event Instance
    An event instance is a single event in a table format.
    The table includes event type and several argument roles together with corresponding arguments.
    The example is shown as below:

    Tom *bought* 2 pounds of flour at Pinshihui for $5 per pound last night.

    +----------+-------------------+
    |      Event Type: Buy         |
    +==========+===================+
    | Buyer    | Tom               |
    +----------+-------------------+
    | Object   | 2 pounds of flour |
    +----------+-------------------+
    | Price    | $5 per pound      |
    +----------+-------------------+
    | Time     | last night        |
    +----------+-------------------+
    | Location | Pinshihui         |
    +----------+-------------------+
    | Cashier  | N/A               |
    +----------+-------------------+

Trigger
    Refering the annotation guide of ACE05 [#ace05]_, event trigger is
    the word that most clearly expresses event's occurrence.
    For instance, the trigger word of the example above is *bought*.

Argument Role
    Argument roles are event participants' types.
    For instance, *Buyer*, *Object*, *Price*, *Time*, *Location* and *Cashier* are argument roles.
    These roles are pre-defined together with event types.
    Each event type correspondes to a specific event template table.

Argument
    Arguments are participants to corresponding roles.
    Arguments can be absent if the context cannot provide the information.
    For example, we don't know who is the cashier when Tom bought flour last night,
    so here the argument to *Cashier* role is N/A.

Combination
    Argument combinations are ``set`` without inner argument orders.
    For example, the combination of the above example is ``{last night, Tom, $5 per pound, 2 pounds of flour, Pinshihui}``.
    N/A is not included in combinations.

Entity & Mention
    Entities are basic elements of objects.
    For example, ``Tom`` is a ``PERSON`` entity.
    One entity may have multiple mentions, and a mention could be
    an occurrence in the raw text, or a pronoun refering to the same entity.

Span
    Span indicates the positions ``[sentence idx, start char idx, end char idx + 1]``.
    For instance, ``[0, 1, 3]`` refers to ``bought 2`` if we apply space tokenisation.


References
----------

.. [#ace05] https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/english-events-guidelines-v5.4.3.pdf
