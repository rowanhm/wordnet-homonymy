# WordNet Homonymy

The repository for our paper, ['Homonymy Information for English WordNet'](http://www.lrec-conf.org/proceedings/lrec2022/workshops/GWLL/pdf/2022.gwll-1.13.pdf).

## Reproduction

To reproduce our work (and produce the homography annotation), enter your OED research API credentials into `data/credentials.csv`, and run the files in `src/stages` sequentially, e.g. starting with `python -m src.stages.s01_extract_wn`. After you are done, remember to remove `data/ox_raw`.

The final data will be saved in the `output` file, as `within_pos_clusters.csv`, `between_pos_clusters.csv`, and `raw_clusters.csv`. Refer to the paper to understand the differences between these.

## Data

We want to release the annotation layer without the need for reproduction, but are waiting for confirmation from the OUP about the release rights for our data. We will update this GitHub as soon as this is confirmed. If it has been a while, do get in touch with us.
