"""FM-Anonymize-Gen: replace FM-Anonymize's blurred heads by synthetic ones.

The generative extension of the paper (Sec. V, version "v7a, full
head"). FM-Anonymize finds and blurs every head; this package turns the
blurred heads into realistic synthetic faces, one image at a time, so
labelers and perception models keep head pose and keypoints while the
identity is gone. Plates, windows, the SAM 3 face net and heads outside
the size gates keep FM-Anonymize's blur.

Per generated head (:func:`blur_data.generate.inpaint.apply_regions`):

1. **Start from the blur.** A square crop with context around the head
   region is cut from the image *as FM-Anonymize blurs it*
   (``blur_scale`` 0.3) and upscaled to 512 px. SD 1.5 inpainting edits
   the grown head region with SDEdit at ``strength`` 0.8: light, colour
   and the hair outline survive the blur, the identity detail does not.
2. **Pose guide.** SAM 3D Body is fitted again on the head's person
   box; a template head (MHR mean shape) is placed at the fitted head
   pose and its depth rendered (depth ControlNet), and the person's own
   face and shoulder keypoints are drawn as an OpenPose map (OpenPose
   ControlNet). The person's own mesh is never rendered
   (:mod:`blur_data.generate.pose_guide`).
3. **Identity.** Every head is conditioned (IP-Adapter-FaceID plus-v2)
   on a synthetic reference face that SD 1.5 paints from a seed alone.
   Heads linked into one track over the frames share the seed, so a
   person keeps one synthetic identity
   (:mod:`blur_data.generate.tracks`,
   :class:`blur_data.models.face_generator.FaceGenerator`).
4. **Composite.** The generated crop is blended back with a Laplacian
   pyramid and matched in blur, noise and JPEG level to its
   surroundings (:mod:`blur_data.generate.harmonize`).

Modules, in pipeline order:

* :mod:`~blur_data.generate.config` - every setting, from
  ``configs/generate.yaml``
* :mod:`~blur_data.generate.routing` - which regions are generated,
  seeds
* :mod:`~blur_data.generate.crops` - square crops, model input, pasting
* :mod:`~blur_data.generate.pose_guide` - template head depth and
  OpenPose map from a SAM 3D Body fit
* :mod:`~blur_data.generate.harmonize` - blend into the scene
* :mod:`~blur_data.generate.inpaint` - one image: generate the heads,
  blur the rest
* :mod:`~blur_data.generate.tracks` - frame sequences, segments, track
  linking
* :mod:`~blur_data.generate.render` - a whole run (used by
  ``blur_data/scripts/blur-data-generate.py``)

The diffusion models live in :mod:`blur_data.models.face_generator`
and :mod:`blur_data.models.faceid`.
"""
