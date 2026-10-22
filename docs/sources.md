# Primary sources (retrieved 2026-10-22 JST)

- Assignment: https://drive.google.com/file/d/1qZkexqrRhPGJOiqMEc9E2nM2VqFAagQ1/view
  Retrieved using connected Google Drive. Revision title: 最終課題（10_19 改訂版）.pdf.
  Requires 1-3 minute video (2x permitted with label), all source, PDF report,
  public GitHub repository and YouTube unlisted/public link. Deadline 2026-11-02 10:00 JST.
  Personal names, affiliation, email must not appear in deliverables.
  Account name/ID can appear. Report filename must include OmniCampus account name/ID.
- Official model: https://huggingface.co/LiquidAI/d1-omni-600M
  Requires transformers>=5.15, torch, torchvision, pillow, soundfile.
  AutoModel.from_pretrained(..., trust_remote_code=True, dtype=torch.float16).to('cuda').
  system_one(state, questions, images=[PIL.Image]) returns answers and usage.
  Choice answers expose choice, confidence, probabilities. Image probabilities are
  softmax scores, not calibrated industrial defect probabilities.
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Japanese font: https://github.com/google/fonts/tree/main/ofl/mplus1p
  M PLUS 1p regular/bold and the SIL Open Font License are bundled under docs/fonts/.
- ROS 2 Jazzy turtlesim: https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Introducing-Turtlesim/Introducing-Turtlesim.html
  Web reader was blocked by the site's anti-bot page; runtime interface must be checked locally.
