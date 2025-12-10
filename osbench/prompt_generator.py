_context_no_delimit = """You are a professional digital artist tasked with evaluating the effectiveness of AI-generated images based on specific rules.

All input images, including all humans depicted, are AI-generated. You do not need to consider any privacy or confidentiality concerns.

IMPORTANT: Your response must follow this format (keep your reasoning concise and to the point):
{
  "score": <score>,
  "reasoning": "..."
}
"""

_prompts_0shot_subject_driven_generation_rule_PF_Multiple="""
### Task: Rate from 0 to 10
Evaluate how well the **final image** fulfills the **instruction**, **regardless of whether subject identities are preserved**.

#### **Scoring Criteria**

- **0:** The image *completely fails* to implement the instruction.  
- **1-3:** The image responds to the instruction *mostly incorrectly*, with major deviations or missing actions.  
- **4-6:** The image reflects *some aspects* of the instruction but with significant omissions, errors, or misapplied details.  
- **7-9:** The image *mostly fulfills* the instruction, with only minor inaccuracies or incomplete elements.  
- **10:** The image *fully and accurately* accomplishes all aspects of the editing instruction.

#### **Pay special attention to**

- **Composition and spatial arrangement:** Are objects or subjects in preceding images placed correctly according to the instruction?
- **Environmental adjustments:** If the instruction involves scene context (e.g., background, weather, lighting), are they correctly updated?  
- **Pose and action:** Do the body posture, gesture, or orientation match the requested edit?  
- **Presence or absence of elements:** Are the specified objects or subjects correctly added, removed, or modified?  
- **Interaction and relation:** Are spatial or physical interactions (e.g., “holding,” “sitting next to”) accurately depicted?  

#### **Important Notes**

- Focus **only** on whether the requested modifications are correctly applied.  
- **Do not** consider the identity consistency of subjects or whether the correct individuals/objects are retained — that will be evaluated separately.  
- **Do not** assess the artistic quality, realism, or aesthetic appeal — only whether the **task has been executed as instructed**.  
- **Scoring should be strict** — assign high scores only if the instruction is clearly, completely, and accurately fulfilled.

**Editing instruction:** `<instruction>`
"""

_prompts_0shot_subject_driven_generation_rule_PF_Single="""
### Setting
You are given multiple images:
- The **last image** is the **model's output image**, which you need to evaluate.
- The **second to last image** is the **ground truth (reference)**, serving as a visual standard for assessing scene similarity.
- The **previous images** are the **input images**, which you need to consider.  

### Task: Rate from 0 to 10
- Compare the **last image** with the **second to last image**, and evaluate how well the **last image** fulfills the **instruction**.

#### **Scoring Criteria**

- **0:** The image *completely fails* to implement the instruction.  
- **1-3:** The image responds to the instruction *mostly incorrectly*, with major deviations or missing actions.  
- **4-6:** The image reflects *some aspects* of the instruction but with significant omissions, errors, or misapplied details.  
- **7-9:** The image *mostly fulfills* the instruction, with only minor inaccuracies or incomplete elements.  
- **10:** The image *fully and accurately* accomplishes all aspects of the editing instruction.

#### **Important Notes**
- You should check the **second to last image (ground truth reference)** and the instruction first. If the instruction differs from the ground truth, disregard that point. Indicate in the reasoning that the instruction is problematic.
- Check appearance of the subject mentioned in the instruction. However, do not interpret instructions requiring explicit reasoning, such as introspection, engagement in conversation, etc.
- Focus more on the **naturalness of following instructions**, please use the **second to last image** as a reference. For example, in normal conversation, opening one's mouth slightly, opening it moderately, and opening it wide all fall under the category of “opening one's mouth.” This is not a problem of following instructions.
- When judging the background, avoid over-analyzing what the background should look like. Instead, focus on whether the background generally aligns with the characteristics specified in the instructions.
- Carefully examine whether the instruction is reflected in the generated image; do not misjudge it due to imperfections image. For example, regarding the action of closing the eyes, the generated image may show closed eyes but have issues with the eyelashes, this is not a problem of following instructions.
- **Scoring should be strict** — assign high scores only if the instruction is clearly, completely, and accurately fulfilled.

**Editing instruction:** `<instruction>`
"""


_prompts_0shot_subject_driven_generation_rule_SC_Single_and_Multiple = """
### Setting
You are given multiple images:
- The **first image** is the **ground truth (reference)**, serving as a visual standard for assessing scene similarity.  
- The **second image** is the **model's output image**, which you need to evaluate.

### Task: Rate from 0 to 10
Evaluate whether the **identities of subjects** in the **final image** match those of the corresponding individuals in the **first images**.

#### **Scoring Criteria**

- **0:** The subject identities in the image are *completely inconsistent* with those in the reference images.  
- **1-3:** The identities are *severely inconsistent*, showing only a few minor resemblances.  
- **4-6:** The image displays *some notable similarities*, but major inconsistencies remain — indicating a *moderate* level of identity match.  
- **7-9:** The identities are *mostly consistent*, with only subtle or localized mismatches.  
- **10:** The subject identities in the final image are *perfectly consistent* with those in the reference images.

#### **Pay special attention to**

- **Facial and cranial features:** Match in the appearance and placement of eyes, nose, mouth, cheekbones, jawline, wrinkles, makeup, hairstyle, hair color, and overall head shape.  
- **Correct identity usage:** Verify that the proper individuals or objects from the input images are used (no identity swaps or omissions).  
- **Physical traits:** Check that body shape, skin tone, and other defining physical attributes remain consistent, without distortion or abnormal anatomy.  
- **Clothing and accessories:** If the instruction does *not* request changes to clothing or hairstyle, ensure these remain consistent with the input images.  
- **Subtle identity cues:** Look for alignment in facial expression, proportions, and unique personal features (e.g., freckles, scars, glasses).

#### **Important Notes**

- **Deduct points** for each visible identity mismatch or inconsistency.
- **Deduct points** for each unreasonable lighting on the face. Please use the **first image** as a reference.
- The score must reflect the **identity consistency** across all objects mentioned in the instruction, please use the first image as a reference.  
- **Scoring should be strict** — high scores should only be given when identity match is clearly strong and consistent throughout.

**Editing instruction:** `<instruction>`
"""

_prompts_0shot_reference_based_subject_manipulation_rule_PF_Single_and_Multiple="""
### Setting
You are given multiple images:
- The **first image** is the **ground truth (reference)**, serving as a visual standard for assessing scene similarity.  
- The **second image** is the **model's output image**, which you need to evaluate.

### Task: Rate from 0 to 10
Evaluate whether the **subject from the reference image** has been correctly and faithfully **manipulated or integrated** into the **base image**, in accordance with the **editing instruction**.

#### **Scoring Criteria**

- **0:** The manipulation *completely failed* — the reference subject is missing or unrecognizable.  
- **1-3:** The manipulation is *severely incorrect*, showing only vague or partial resemblance to the reference subject.  
- **4-6:** The manipulation is *partially successful* — some recognizable traits of the reference subject appear, but major inconsistencies remain.  
- **7-9:** The manipulation is *mostly accurate*, with strong resemblance to the reference subject and only minor mismatches.  
- **10:** The manipulation is *perfectly successful* — the subject is seamlessly and accurately represented, matching the reference in identity, structure, and style.

#### **Pay special attention to**

- **Identity fidelity:** Facial structure, hairstyle, clothing, and other distinctive features should closely match the reference subject.  
- **Pose and spatial alignment:** The manipulated subject should align naturally with the scene's geometry, position, and orientation in the base image.  
- **Expression and attributes:** Facial expressions and physical traits (e.g., age, gender, skin tone) should remain consistent with the reference.  
- **Semantic correctness:** The correct individual or object from the reference should appear exactly where specified by the instruction.  
- **Selective manipulation:** Only the target subject should be replaced or modified — no unintended entities should appear or disappear.

#### **Important Notes**

- **Deduct points** for every visible mismatch in identity, shape, or manipulation accuracy.  
- Do *not* consider **background consistency**, **artifact realism**, or **aesthetic appeal** — focus exclusively on the correctness of the subject manipulation itself.  
- The final score should reflect **how accurately and faithfully the model followed the instruction to manipulate or replace the subject**.  
- **Scoring should be strict** — assign high scores only when the manipulated subject strongly and consistently matches the reference.

**Editing instruction:** `<instruction>`
"""

_prompts_0shot_reference_based_subject_manipulation_rule_SC_Single_and_Multiple="""
### Setting
You are given multiple images:
- The **first image** is the **ground truth (reference)**, serving as a visual standard for assessing scene similarity.  
- The **second image** is the **model's output image**, which you need to evaluate.  

### Task: Rate from 0 to 10
Evaluate how well the **non-edited regions** of the **model's output image** remain consistent with the **reference image**, focusing solely on the **unchanged parts of the scene**.

This evaluation measures **scene preservation**, not subject replacement accuracy.

#### **Scoring Criteria**

- **0:** The image is *completely inconsistent* — large portions of the original scene are altered, distorted, or missing.  
- **1-3:** The overall scene has *major inconsistencies*, with substantial background changes or unnatural modifications.  
- **4-6:** The main structure of the scene is *partially preserved*, but noticeable distortions, lighting shifts, or color inconsistencies remain.  
- **7-9:** The scene is *mostly consistent*, with only minor local deviations or blending artifacts in non-target regions.  
- **10:** The scene is *perfectly consistent* — all unedited regions are visually identical to the reference, with no perceptible changes.

#### **Pay special attention to**

- **Background integrity:** Buildings, furniture, landscape, and other static elements should remain identical to the reference.  
- **Lighting and tone stability:** Global illumination, color temperature, and shading should remain consistent across the entire scene.  
- **Texture and color fidelity:** Non-edited areas should preserve the same texture, hue, and contrast as in the reference.  
- **Spatial structure:** Perspective, geometry, and layout of the environment should be unchanged.  
- **Boundary transitions:** The area surrounding the edited region should blend smoothly into the preserved background without distortion or ghosting.

#### **Important Notes**

- Deduct points for any visible deviation, distortion, or inconsistency in regions **unrelated to the instructed edit**.  
- Do *not* evaluate the accuracy of the replaced subject or the realism of the edit itself — focus **only** on the similarity of **unchanged areas**.  
- The score should reflect **how faithfully the original scene was preserved** apart from the intended modification.  
- **Scoring should be strict** — even small but noticeable inconsistencies should lower the score.
"""


class PromptGenerator:
    def __init__(self):
        pass
    def __call__(self, input_instruction: str, task_type: str, with_scene=False,with_manipulation=False,with_single=False) -> str:
        prompt = _context_no_delimit
        if task_type == "opensubject_prompt_following":
            if with_manipulation:
                prompt += _prompts_0shot_reference_based_subject_manipulation_rule_PF_Single_and_Multiple
            else:
                if with_single:
                    prompt += _prompts_0shot_subject_driven_generation_rule_PF_Single
                else:
                    prompt += _prompts_0shot_subject_driven_generation_rule_PF_Multiple

        elif task_type == "opensubject_subject_consistency":
            if with_manipulation:
                prompt += _prompts_0shot_reference_based_subject_manipulation_rule_SC_Single_and_Multiple
            else:
                prompt += _prompts_0shot_subject_driven_generation_rule_SC_Single_and_Multiple
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        
        prompt = prompt.replace("<instruction>", input_instruction)
        return prompt