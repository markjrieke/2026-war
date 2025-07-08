# DEVLOG

## 2025-07-04

I started to get into some more dev model work this past week, but realized I needed to make some more data updates in order to continue. Some of these I may have been able to write a scraper or interface to some API to automate a bit, but there are so many edge cases (and the source data from the sites I'd scrape sometimes is self-conflicting) that I feel better just going through it manually. Anyway, this past week has been wholly focused on incumbents:

* Handling the coding of the incumbent parties when a seat was vacant (i.e., what was the party of the candidate *prior to* the vacancy).
* Handling third party incumbents who caucus with either the democrats or republicans (e.g., Bernie Sanders).
* Handling cases when the incumbent party *is not* the winning party from the previous election. The majority of the time, this is just due to special elections in between generals, so is fine, but I did have to swap a few incorrectly encoded incumbent parties around.
* Adding indicators for whether (or not) the incumbent candidate is running.

The last item is necessary for how (I'm thinking) I want to write the model. In addition to including an `incumbent_running` parameter, I'm thinking of modeling the incumbent party's two-party voteshare margin as a function of candidates. I'll likely continue this last item bit by bit over the next week.

## 2025-06-27

Alrighty, I've adjusted the house dataset to address the initial tranche of nuances --- mostly around dealing with jungle primaries vs. general elections. Ideally next week I get into some basic model building, but we'll see what crops up!

## 2025-06-20

Data validation is the name of the game this week. Only thing I did was generate a mapping table that mapped candidate names/politician ids to races. Next week, I need to start addressing the [nuances](https://github.com/markjrieke/2026-war/issues/2) that I found while going through the dataset.

## 2025-06-13

Much like the [March Madness project](https://github.com/markjrieke/2025-march-madness) I worked on earlier this year, I'm starting a devlog for this project on estimating candidate quality via a Wins Above Replacement (WAR)-esque metric. Again, I expect that this will be beneficial to future viewers of this project, including myself.

There are four basic elements of this project:

* Validating/adjusting the data
* Building the model
* Generating publication-quality graphics
* Writing explainer article(s)

I'm working from an existing dataset --- thankfully, that means that I won't need to spend a lot of time sourcing the data. But I will need to go through and validate/adjust so that it'll work with the model formulation I have in mind. Oddly enough, the model that I have in my head is *incredibly similar* to the March Madness model, with a few tweaks to be able to estimate WAR.[^1]

I don't have a strict timeline for this project (well, I suppose I need to have it done ahead of the 2026 midterms, but those are *lightyears* away at the moment). Ideally around November, I guess? Anyway, this week, I setup a super minimal dev model and put together some of the package infrastructure.

[^1]: There's this weird throughline in which a [model I put together for *Um, Actually*](https://www.thedatadiary.net/posts/2024-10-06-actually/) was a pretty solid base for building a March Madness model, which in turn ends up being a pretty solid base for this model. Just a weird set of connections, man!
