import annotated_images

# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
annotated_images.split("/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/DEArt/*", output_dir='/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/DEArt_split', seed=1337, ratio=(.7, .15, .15))


