# Contributing to momepy

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the this document for
different ways to help and details about how this project handles them. Please make sure
to read the relevant section before making your contribution. It will make it a lot
easier for us maintainers and smooth out the experience for all involved. The community
looks forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine.
> There are other easy ways to support the project and show your appreciation, which we
> would also be very happy about: - Star the project - Tweet about it - Refer this
> project in your project's readme - Mention the project at local meetups and tell your
> friends/colleagues

## I Have a Question

> If you want to ask a question, we assume that you have read the available
> [Documentation](https://pysal.org/momepy).

Before you ask a question, it is best to search for existing
[Issues](https:/github.com/pysal/momepy/issues) that might help you. In case you have
found a suitable issue and still need clarification, you can write your question in this
issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend
the following:

- Open an [Issue](https:/github.com/pysal/momepy/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions, depending on what seems relevant.

We will then take care of the issue as soon as possible.

### Discord

You may also want to join [PySAL Discord server](https://discord.gg/BxFTEPFFZn) and ask
your question there. Just note, that Discord is primarily for ephemeral developer
discussion and every question others may benefit from shall be asked publicly on GitHub.

### Reporting Bugs

<!-- omit in toc -->
#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information.
Therefore, we ask you to investigate carefully, collect information and describe the
issue in detail in your report. Please complete the following steps in advance to help
us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using
  incompatible environment components/versions (Make sure that you have read the
  [documentation](https://pysal.org/momepy).)
- To see if other users have experienced (and potentially already solved) the same issue
  you are having, check if there is not already a bug report existing for your bug or
  error in the [bug tracker](https:/github.com/pysal/momepy/issues?q=label%3Abug).
- Also make sure to search the internet (including Stack Overflow) to see if users
  outside of the GitHub community have discussed the issue.
- Collect information about the bug:
- Stack trace (Traceback)
- OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
- Version of the Python and all the dependencies.
- Possibly your input and the output
- Can you reliably reproduce the issue? And can you also reproduce it with older
  versions?

<!-- omit in toc -->
#### How Do I Submit a Good Bug Report?

We use GitHub issues to track bugs and errors. If you run into an issue with the
project:

- Open an [Issue](https:/github.com/pysal/momepy/issues/new). (Since we can't be sure
  at this point whether it is a bug or not, we ask you not to talk about a bug yet and
  not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that
  someone else can follow to recreate the issue on their own. This usually includes your
  code. For good bug reports you should isolate the problem and create a reduced test
  case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are
  no reproduction steps or no obvious way to reproduce the issue, the team will ask you
  for those steps. Bugs that cannot be reproduced will not be addressed until they are
  reproduced.
- If the team is able to reproduce the issue, it will be marked, and the issue will be
  left to be implemented by someone.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for momepy,
**including completely new features and minor improvements to existing functionality**.
Following these guidelines will help maintainers and the community to understand your
suggestion and find related suggestions.

<!-- omit in toc -->
#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://pysal.org/momepy) carefully and find out if the
  functionality is already covered, maybe by an individual configuration.
- Perform a [search](https:/github.com/pysal/momepy/issues) to see if the enhancement
  has already been suggested. If it has, add a comment to the existing issue instead of
  opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you
  to make a strong case to convince the project's developers of the merits of this
  feature. Keep in mind that we want features that will be useful to the majority of our
  users and not just a small subset. If you're just targeting a minority of users,
  consider writing an add-on/plugin library.

<!-- omit in toc -->
#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as
[GitHub issues](https:/github.com/pysal/momepy/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details
  as possible.
- **Describe the current behavior** and **explain which behavior you expected to see
  instead** and why. At this point you can also tell which alternatives do not work for
  you.
- **Explain why this enhancement would be useful** to most momepy users. You may also
  want to point out the other projects that solved it better and which could serve as
  inspiration.

### Code Contribution

You can create a development environment using conda using the environment files with `latest` in the `ci` folder:

```sh
conda env create -f ci/envs/314-latest.yaml -n momepy
```

To install `momepy` to the Conda environment in an editable form, clone the repository,
navigate to the main directory and install it with pip:

```sh
pip install -e .
```

When submitting a pull request:

- All existing tests should pass. Please make sure that the test suite passes, both
  locally and on GitHub Actions. Status on GHA will be visible on a pull request. GHA
  are automatically enabled on your own fork as well. To trigger a check, make a PR to
  your own fork.
- Ensure that documentation has built correctly. It will be automatically built after
  merging your commit to main.
- New functionality ***must*** include tests. Please write reasonable tests for your
  code and make sure that they pass on your pull request.
- Classes, methods, functions, etc. should have docstrings. The first line of a
  docstring should be a standalone summary. Parameters and return values should be
  documented explicitly.
- Follow PEP 8 when possible. We use `ruff` for linting and formatting to ensure
  robustness & consistency in code throughout the project. Ruff is included in the `pre-commit` hook and style will
  be checked on every PR.
- `momepy` supports Python versions according to
  [SPEC0](https://scientific-python.org/specs/spec-0000/). When possible, do not
  introduce additional dependencies. If that is necessary, make sure they can be treated
  as optional.

#### Procedure

1. *After* opening an issue and discussing with the development team, create a PR with
   the proposed changes.

<!-- omit in toc -->
## Attribution

This guide is based on the **contributing-gen**.
[Make your own](https://github.com/bttger/contributing-gen)!
