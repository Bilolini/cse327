# Third-Party Attribution

This project adapts existing soccer analytics tools. Its contributions concern integration, data conversion, compatibility fixes, and evaluation. The detectors, ByteTrack, SigLIP, UMAP, KMeans, and GTA-link were developed by their respective authors.

| Component | Source | Usage |
| --- | --- | --- |
| Football AI notebook and sports utilities | https://github.com/roboflow/sports | Foundation for the published vision notebook |
| Tracking and event workflow | https://github.com/JohnComonitski/FootballTrackingDataGeneration and https://github.com/JohnComonitski/TrackingDataToEventData | Adapted tracking-to-event workflow and event-generation code |
| FootballMatchAnalysis | https://github.com/JohnComonitski/FootballMatchAnalysis | Bundled analysis dependency; original MIT license retained at `TrackingDataToEventData/FootballMatchAnalysis/LICENSE` |
| Metrica analysis routines | https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking | IO, visualization, velocities, and pitch-control code in the bundled dependency; source headers retained |
| ByteTrack | https://github.com/ifzhang/ByteTrack | Tracking through Supervision |
| GTA-link | https://github.com/sjc042/gta-link | Optional external tracklet refinement; implementation and weights not bundled |
| Metrica sample data | https://github.com/metrica-sports/sample-data | Reference format and evaluation inputs; full inputs not bundled |

Existing copyright and attribution notices in copied source files are preserved. No blanket license is asserted over all upstream code or data. Consult the relevant upstream terms before reuse. Pretrained weights and match footage must be acquired separately under their own terms.

The CSVs in `evaluation/` are saved project-development summaries. They evaluate the event stage and a tracklet ablation; they do not establish end-to-end video event accuracy.
