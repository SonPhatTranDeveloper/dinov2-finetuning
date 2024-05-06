from vision_sinkformer import vit_small

if __name__ == "__main__":
    # Load pretrained model
    model = vit_small(patch_size=14,
                      img_size=526,
                      init_values=1.0,
                      num_register_tokens=4,
                      block_chunks=0)

    # Load the weights

    # View the layers in the model
    print(model)

