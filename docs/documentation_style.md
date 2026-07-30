# Documentation style

The audience is someone who has found a function and wants to know what it does.
They can read the code. What they cannot recover from the code is the intent, the
contract, and the mathematics the code is an encoding of. Write that.

## What gets documented

Public classes, functions and methods get a docstring.

Private helpers do not need one. When a private helper solves a self contained
mathematical problem, give it a docstring that states the problem, because that
is the one thing a reader cannot infer. Everything else about a private helper
belongs in comments.

There are no module level docstrings. Nobody reads them and they are one more
place to keep in sync. Content that would have gone in one goes into a design
note block instead.

## Structure

Numpy style, with these sections and no others.

```
Parameters
Returns
Raises
Attributes
References
Notes
```

`Raises` and `Attributes` earn their place. Error behaviour is part of the
contract, and a state object whose fields are its interface is better documented
by naming them than by prose. Nothing else gets a section heading.

Give shape and dtype for every tensor, in the batched form the function actually
takes, such as `(N, n)` or `(N, m, m)`. State which entries may be `None` and
when.

Say how the arguments and results relate to one another. A function returning a
vector and a metric must say whether the vector lives in the space that metric
measures. A function returning several tensors must say which index of one lines
up with which index of another. These relationships are the most common source
of misuse and they are invisible in the signature.

## What to say

Say what the function computes, in terms a caller can act on.

Include the mathematics that defines the result. A solver states the exact
equation it solves. An energy states the energy. Define every symbol you use.
Prefer the equation over a sentence describing the equation.

Say what raises and under what condition. Error behaviour is part of the
contract, not an implementation detail, so it stays in the docstring even though
it looks like a detail about the code.

Name the precondition of any guarantee. A property that holds only for small
step sizes, only up to a solver tolerance, or only when the caller has already
called something else, is not the same property as one that always holds, and a
reader will assume the stronger reading unless told.

## What not to say

No description of how the result is computed. Which factorization, which loop,
which order of operations, which tensor is reused. That is the code's job.

No development history and no forward looking notes. Nothing that reads as "not
done yet", "for now", "a later version will", or a reference to a discussion, a
review or a commit. If it is not true of the code as it stands, it does not go in
the documentation.

No restating of the signature in prose.

## Do not compare against other algorithms

Say what this code does. Do not explain it by contrast with a sibling
implementation, as in "unlike X, this one ...". The reader may not know X, X may
be a niche method nobody would accept as the reference point, and the sentence
goes stale the moment X changes.

The one comparison worth making is against the base class, where a subclass
departs from documented shared behaviour. That is a contract the reader is
already holding.

A general-purpose tool must not need a specific algorithm to explain itself. If a
root finder can only be described through the equation one particular sampler
hands it, either the description is wrong or the tool is not general.

## Notation

The library has one vocabulary and every file uses it. Position is `q`, momentum
is `p`, inverse temperature is `beta`. A file whose source material uses other
letters is translated into the library's names rather than carrying its own.

Define a symbol before using it. A formula that mentions a matrix the reader has
not been introduced to is not documentation.

## Design notes and derivations

A comment block delimited by full width `#` rules is for what no docstring should
carry: things that are about more than one API surface, or about why the code took
the shape it did.

It belongs there when it is

* how two things relate, such as a sampler and the state object it threads,
* a derivation of an equation the code encodes,
* a reparameterisation that makes a solve tractable or cheap,
* a trade off taken deliberately, together with what it costs.

It does not belong there when it is

* the defining property of a class or function, including its equation. A solver
  implementing Newton says so in its docstring and states the Newton step there.
  In a design note it is hidden from the reader who found the class.
* the contract of an argument or a return value. Shapes, admissible values,
  what raises, which rule needs what from its caller.

A derivation may be as long as it needs to be, in the block or in a comment at
the code that performs it. Showing the work is the point, so length is not a
fault there. The test is whether a reader who has the docstring in front of them
still needs it.

## Comments

An inline comment is at most two lines. The single exception is a mathematical
derivation, which may run as long as the mathematics requires.

Use a comment for the things a docstring must not carry. Why this factorization
and not another, why a guard is present, which invariant the next few lines rely
on.

## Language

Plain language and complete sentences. Do not use a semicolon and do not use a
dash as punctuation. Break the sentence in two instead.

Prefer the direct statement. "Raises when the covariance is not positive
definite" rather than "note that it should be pointed out that an error may be
raised in the event that".

## Keeping it true

Every claim in a docstring is a claim the code must honour. A named term must
appear in the code. A stated guarantee must hold, or must state its precondition.
A documented default must be the default.

When an invariant changes, delete what documented the old one. That applies to
tests as well. A test that pins an invariant the design has moved past is not
protecting anything, and reading it as a constraint on new work is worse than
having no test at all.

## Size

Documentation competes with the code for the reader's attention, so its size is
part of the design. A useful check is the ratio of documentation lines to code
lines, per file. It is a ceiling, not a target: most files here sit between 0.5
and 1.5 to 1, and a low ratio is not a defect. Much above 1.5 is a file claiming
to be unusually subtle, and that claim is usually wrong.

The same applies to a design note. Two hundred lines of note above two hundred
lines of code means the note is arguing rather than stating. Keep the
derivation, the identity, the caveat with teeth. Cut the paragraph that restates
the equation above it in prose, and cut the measurement whose source no reader
can find.
