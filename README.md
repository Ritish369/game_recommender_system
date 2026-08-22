# Game Recommender System

*tl;dr: dropping user_id beat the full model on Test Recall. See why below.*

## Results

| Method | Test Recall@10 |
|---|---|
| **v1 (no_user_id)** | **0.0812** |
| v1 (full) | 0.0678 |
| Popularity baseline | 0.0395 |
| Random expected | 0.0050 |

Removing user information from the model actually improved recall, a sign it was overfitting to individual users rather than learning general preferences from a small user pool.

- [v0](https://github.com/Ritish369/game_recommender_system/tree/main/v0), the starting point (MVP)
- [v1](https://github.com/Ritish369/game_recommender_system/tree/main/v1), resolves v0's issues: adds semantic understanding between items, users, and their interactions, and reduces overfitting on a small user pool

This project is paused right now, but I'm looking forward to completing it soon. Any feedback is appreciated, do discuss.

This repository supersedes an earlier, more exploratory pass at the same problem: [game_recsys](https://github.com/Ritish369/game_recsys), from right after finishing Kim Falk's book. This is the repository to look at.

<details>
<summary><b>The full story: from a thought, through a PRD, to v1</b></summary>

From a thought/idea, through PRD creation, system design, to implementation and iteration.

The goal of this project was to learn about and understand recommendation systems in the most unusual, self-paced, and ridiculously stressful way possible. Hence, I did the same, apparently.

The complete list of references can be found [here](https://docs.google.com/document/d/12JNdq_meOgBmCefplIhcSE0LaagRyTavRKdVhQYKMlI/edit?usp=sharing).

The project started by thinking about the requirements, the problems being faced by the products in this domain, and what minimalist solution could be provided to them. All this was consolidated in a Product Requirements Document (PRD), on pen and paper.

One of the foremost problems I faced personally was that gaming websites are very overwhelming: too wordy, cluttered, and similar to one another. So, I wanted to build a minimalistic version of such a system. And, woah! I just had the thought that this might be a UI, or related, problem.

Anyways! This has been started.

After building the PRD, came the designing of the system. For the system design thought process, I read some articles, watched videos, and similar material, as listed in the references, and therefore built the complete system design found [here](https://github.com/Ritish369/game_recommender_system/blob/main/images/recommender_system_design.png). It was built by hand first, then finalised in this format.

Eventually, v0 (the MVP) was built.

**V0**: It is the starting point, the first implementation. More details can be found [here](https://github.com/Ritish369/game_recommender_system/tree/main/v0).

**V1**: It is the improved version of v0, with a focus on resolving the issues observed in v0's results. One of those issues was no semantic understanding between items, user behaviour, and their interactions, which limited its ability to provide personalised recommendations. Another one was overfitting to user data, given the small user pool. More details can be found [here](https://github.com/Ritish369/game_recommender_system/tree/main/v1).

The implementation and iterations gave me a deep understanding of recommenders, especially of contrastive learning. One of the problems observed through v1 was that the no_user_id ablation gave the best result in terms of Test Recall. This means that when user information is not considered, the model actually gives better recommendations, implying that including user information was causing it to overfit rather than actually learn about the user. Therefore, one solution is more data, i.e., more users in the pool.

</details>
