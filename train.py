"""Module containing training functions for the various models evaluated in the DECAF paper."""

import os
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tensorflow as tf
import torch
from sklearn.neural_network import MLPClassifier

from data import DataModule
from models.DECAF import DECAF

models_dir = "weights"


def train_vanilla_gan(
    train_dataset,
    noise_dim=32,
    dim=128,
    batch_size=128,
    log_step=100,
    epochs=50,
    learning_rate=5e-4,
    beta_1=0.5,
    beta_2=0.9,
    model_name="vanilla_gan",
):

    model = VanilllaGAN
    model_filename = os.path.join(models_dir, f"{model_name}.pkl")

    num_cols = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = [
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "income",
    ]

    gan_args = ModelParameters(
        batch_size=batch_size,
        lr=learning_rate,
        betas=(beta_1, beta_2),
        noise_dim=noise_dim,
        layers_dim=dim,
    )

    train_args = TrainParameters(epochs=epochs, sample_interval=log_step)

    synthesizer = model(gan_args)

    if os.path.exists(model_filename):
        synthesizer = model.load(model_filename)
    else:
        synthesizer.train(
            data=train_dataset,
            train_arguments=train_args,
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        synthesizer.save(model_filename)

    synth_dataset = synthesizer.sample(len(train_dataset))

    return synth_dataset


def train_wgan_gp(
    train_dataset,
    noise_dim=128,
    dim=128,
    batch_size=500,
    log_step=100,
    epochs=50,
    learning_rate=[5e-4, 3e-3],
    beta_1=0.5,
    beta_2=0.9,
    model_name="wgan_gp",
):
    model = WGAN_GP
    model_filename = os.path.join(models_dir, f"{model_name}.pkl")

    num_cols = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = [
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "income",
    ]

    gan_args = ModelParameters(
        batch_size=batch_size,
        lr=learning_rate,
        betas=(beta_1, beta_2),
        noise_dim=noise_dim,
        layers_dim=dim,
    )

    train_args = TrainParameters(epochs=epochs, sample_interval=log_step)

    synthesizer = model(gan_args, n_critic=2)

    if os.path.exists(model_filename):
        synthesizer = model.load(model_filename)
    else:
        synthesizer.train(train_dataset, train_args, num_cols, cat_cols)
        synthesizer.save(model_filename)

    synth_dataset = synthesizer.sample(len(train_dataset))

    return synth_dataset


def train_fairgan(
    train_dataset,
    embedding_dim=128,
    random_dim=128,
    generator_dims=(128, 128),
    discriminator_dims=(128, 128),
    bn_decay=0.99,
    l2_scale=0.001,
    batch_size=100,
    pretrain_epochs=50,
    train_epochs=10,
    model_name="fairgan",
):
    tf.compat.v1.disable_eager_execution()

    data = train_dataset.values

    data_filename = os.path.join(models_dir, "adult.npy")

    with open(data_filename, "wb") as data_file:
        pickle.dump(data, data_file)

    inputDim = data.shape[1] - 1
    inputNum = data.shape[0]
    tf.compat.v1.reset_default_graph()
    mg = Medgan(
        dataType="count",
        inputDim=inputDim,
        embeddingDim=embedding_dim,
        randomDim=random_dim,
        generatorDims=generator_dims,
        discriminatorDims=discriminator_dims,
        compressDims=(),
        decompressDims=(),
        bnDecay=bn_decay,
        l2scale=l2_scale,
    )

    out_file = os.path.join(models_dir, model_name)

    if not os.path.exists(out_file + ".meta"):
        mg.train(
            dataPath=data_filename,
            modelPath="",
            outPath=out_file,
            pretrainEpochs=pretrain_epochs,
            nEpochs=train_epochs,
            discriminatorTrainPeriod=2,
            generatorTrainPeriod=1,
            pretrainBatchSize=batch_size,
            batchSize=batch_size,
            saveMaxKeep=0,
        )

    tf.compat.v1.reset_default_graph()
    synth_data = mg.generateData(
        nSamples=inputNum, modelFile=out_file, batchSize=batch_size, outFile=out_file
    )

    mlp = MLPClassifier()
    X_train, y_train = train_dataset.drop(columns=["income"]), train_dataset["income"]
    mlp.fit(X_train, y_train)
    income = mlp.predict(synth_data)
    synth_data = np.append(synth_data, income.reshape((len(income), 1)), axis=1)

    return pd.DataFrame(synth_data, columns=train_dataset.columns)


def train_decaf(
    train_dataset,
    dag_seed,
    h_dim=100,
    lr=0.5e-3,
    batch_size=128,
    lambda_privacy=0,
    lambda_gp=10,
    d_updates=10,
    alpha=2,
    rho=2,
    weight_decay=1e-2,
    grad_dag_loss=False,
    l1_g=0,
    l1_W=1e-4,
    p_gen=-1,
    use_mask=True,
    epochs=50,
    model_name="decaf",
):

    if model_name:
        model_filename = os.path.join(models_dir, f"{model_name}.pkl")
        if os.path.exists(model_filename):
            print("Model path already existed. Loading from {}".format(model_filename))
            return torch.load(model_filename)

    dm = DataModule(train_dataset.values)

    model = DECAF(
        dm.dims[0],
        dag_seed=dag_seed,
        h_dim=h_dim,
        lr=lr,
        batch_size=batch_size,
        lambda_privacy=lambda_privacy,
        lambda_gp=lambda_gp,
        d_updates=d_updates,
        alpha=alpha,
        rho=rho,
        weight_decay=weight_decay,
        grad_dag_loss=grad_dag_loss,
        l1_g=l1_g,
        l1_W=l1_W,
        p_gen=p_gen,
        use_mask=use_mask,
    )

    trainer = pl.Trainer(max_epochs=epochs, logger=False)
    trainer.fit(model, dm)

    if model_name:
        torch.save(model, model_filename)

    return model


def gen_decaf(model, train_dataset, biased_edges):
    if type(model) == str:
        model = torch.load(model)
    dm = DataModule(train_dataset.values)

    # Generate synthetic data
    synth_dataset = (
        model.gen_synthetic(
            dm.dataset.x,
            gen_order=model.get_gen_order(),
            biased_edges=biased_edges,
        )
        .detach()
        .numpy()
    )
    synth_dataset[:, -1] = synth_dataset[:, -1].astype(np.int8)

    synth_dataset = pd.DataFrame(
        synth_dataset, index=train_dataset.index, columns=train_dataset.columns
    )

    if "approved" in synth_dataset.columns:
        # Binarise columns for credit dataset
        synth_dataset["ethnicity"] = np.round(synth_dataset["ethnicity"])
        synth_dataset["approved"] = np.round(synth_dataset["approved"])
    else:
        # Binarise columns for adult dataset
        synth_dataset["sex"] = np.round(synth_dataset["sex"])
        synth_dataset["income"] = np.round(synth_dataset["income"])

    return synth_dataset
