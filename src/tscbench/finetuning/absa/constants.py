# -*- coding: utf-8 -*-
""" Constants and dictionnaries for the project

@Author: Evan Dufraisse
@Date: Fri Nov 25 2022
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2022 CEA - LASTI
"""
import tscbench.modeling.models.absa.prompt as pm
import tscbench.modeling.models.absa.td as td
import tscbench.modeling.models.absa.spc as spc
import tscbench.modeling.models.absa.zero_shot as zs
import tscbench.modeling.models.absa.soft_prompt_cls as spm
import tscbench.modeling.models.absa.no_mention as nom
import tscbench.modeling.models.absa.soft_cls as sc
import json

ABSA_MODELS_CONFIGURATORS = {
    "pm": pm.get_model,
    "td": td.get_model,
    "spc": spc.get_model,
    "zs": zs.get_model,
    # "spm": spm.get_model
}

ABSA_MODELS_BUILDER = {
    "pm": pm.PromptModel,
    "td": td.TdModel,
    "spc": spc.SpcModel,
    "zs": zs.ZeroShotModel,
    "spm": spm.SoftPromptModel,
    "nom": nom.NoMentionModel,
    "sc": sc.SoftCls,
}

MODE_MASK = {
    "pm": True,
    "td": False,
    "spc": False,
    "zs": True,
    "spm": True,
    "nom": False,
    "sc": False,
}


DATASET_CONFIGS = {
    "newsmtsc4splitindep_test_valdependent43": json.loads(
        """{
	"name_dataset":"newsmtsc",
	"format":"newsmtsc",
	"folder_dataset":"newsmtsc4splitdep_test_valindependent43",
	"filenames":{"train":"train.jsonl", "test":"test_dep.jsonl", "validation":"val_dep.jsonl"},
	"splitting_strategy":{"train":[1,0,0], "test":[0,1,0], "validation":[0,0,1]}
}"""
    ),
    "newsmtsc4splitindep_test_valindependent43": json.loads(
        """{
	"name_dataset":"newsmtsc",
	"format":"newsmtsc",
	"folder_dataset":"newsmtsc4splitindep_test_valindependent43",
	"filenames":{"train":"train.jsonl", "test":"test_indep.jsonl", "validation":"validation.jsonl"},
	"splitting_strategy":{"train":[1,0,0], "test":[0,1,0], "validation":[0,0,1]}
}"""
    ),
    "newsmtscmt": json.loads(
        """{
	"name_dataset":"newsmtscmt",
	"format":"newsmtsc",
	"folder_dataset":"newsmtscmt",
	"filenames":{"train":"train.jsonl", "test":"devtest_mt.jsonl"},
	"splitting_strategy":{"train":[1,0,0], "test":[0,0.7,0.3]}
} 
"""
    ),
}


# Problème coréférences
# Problème de négation
# Problème de proximité de la prompt à des adjectifs positifs
# Variations des biais sur des entités du dataset

TEMPLATES_NEWSMTSC_ = {
    "negation": {},
    "coreference": {},
    "proximity": {},
    "bias": {},
}

ENTITIES_NEWSMTSC = [
    {"entity": "Robert", "pron": "he", "poss": "him"},
    {"entity": "John", "pron": "he", "poss": "him"},
    {"entity": "Mary", "pron": "she", "poss": "her"},
    {"entity": "Patricia", "pron": "she", "poss": "her"},
]


TEMPLATES_NEWSMTSC = [
    {
        "id": 0,
        "template": "Contrary to {entity1}, I hate {entity2}.",  # Test de la compréhension du modèle
        "sentiments": [4.0, 2.0],
        "bias": "contrary_to negation",
        "rank_preference": [1, 2],
        "partner": [],
    },
    {
        "id": 1,
        "template": "I hate {entity1} and {entity2}.",  # Test de la compréhension de la conjonction par le modèle, et de l'impact de la proximité de la prompt
        "sentiments": [2.0, 2.0],
        "bias": "conjonction_proximity",
        "rank_preference": [1, 1],
        "partner": [1, 2],
        "rank_preference_partner": {1: [2, 2], 2: [1, 1]},
    },
    {
        "id": 2,
        "template": "I love {entity1} and {entity2}.",  # Test de la compréhension de la conjonction par le modèle, et de l'impact de la proximité de la prompt
        "sentiments": [6.0, 6.0],
        "bias": "conjonction_proximity",
        "rank_preference": [1, 1],
        "partner": [1, 2],
    },
    {
        "id": 3,
        "template": "I love {entity1} but I hate {entity2}.",  # Test de la compréhension de l'opposition par le modèle, et de l'impact de la proximité de la prompt
        "sentiments": [6.0, 2.0],
        "bias": "opposition_proximity",
        "rank_preference": [1, 2],
        "partner": [3, 4],
        "rank_preference_partner": {3: [1, 2], 4: [2, 1]},
    },
    {
        "id": 4,
        "template": "I hate {entity1} but I love {entity2}.",
        "sentiments": [2.0, 6.0],
        "bias": "opposition_proximity",
        "rank_preference": [2, 1],
        "partner": [3, 4],
    },
    {
        "id": 5,
        "template": "I'm far from loving {entity1}",
        "sentiments": [2.0],
        "bias": "far_from_negation",
        "rank_preference": [1],
        "partner": [5, 6],
        "rank_preference_partner": {5: [2], 6: [1]},
    },
    {
        "id": 6,
        "template": "I'm far from hating {entity1}",
        "sentiments": [4.0],
        "bias": "far_from_negation",
        "rank_preference": [1],
        "partner": [5, 6],
    },
    {
        "id": 7,
        "template": "I don't hate {entity1}",
        "sentiments": [4.0],
        "bias": "negation",
        "rank_preference": [1],
        "partner": [8],
        "rank_preference_partner": {7: [1], 8: [1]},
    },
    {
        "id": 8,
        "template": "I don't love {entity1}",
        "sentiments": [4.0],
        "bias": "negation",
        "rank_preference": [1],
        "partner": [7],
    },
    {
        "id": 9,
        "template": "I like {entity1} but I don't like {entity2}",
        "sentiments": [6.0, 2.0],
        "bias": "proximity",
        "rank_preference": [1, 2],
        "partner": [],
    },
    {
        "id": 10,
        "template": "I would like {entity1} if {pron1} was kinder.",
        "sentiments": [2.0],
        "bias": "gramatical",
        "rank_preference": [1],
        "partner": [],
    },
    {
        "id": 11,
        "template": "I'm so proud of {entity1}",
        "sentiments": [6.0],
        "bias": "gramatical",
        "rank_preference": [1],
        "partner": [],
    },
    {
        "id": 12,
        "template": "{entity1} would definitely won a debate against {entity2}",
        "sentiments": [6.0, 2.0],
        "bias": "proximity",
        "rank_preference": [1, 2],
        "partner": [12, 13],
        "rank_preference_partner": {12: [1, 2], 13: [2, 1]},
    },
    {
        "id": 13,
        "template": "{entity1} was ridicule compared to {entity2}",
        "sentiments": [2.0, 6.0],
        "bias": "proximity",
        "rank_preference": [2, 1],
        "partner": [12, 13],
    },
    {
        "id": 14,
        "template": "{entity1} was ridicule facing {entity2}",
        "sentiments": [2.0, 6.0],
        "bias": "proximity",
        "rank_preference": [2, 1],
        "partner": [14, 15],
        "rank_preference_partner": {14: [2, 1], 15: [2, 1]},
    },
    {
        "id": 15,
        "template": "{entity1} was ridicule against {entity2}",
        "sentiments": [2.0, 6.0],
        "bias": "proximity",
        "rank_preference": [2, 1],
        "partner": [14, 15],
    },
    {
        "id": 16,
        "template": "I love {entity1}. I hate {entity2}. {entity1} has a blue shirt.",
        "sentiments": [6.0, 2.0],
        "bias": "proximity",
        "rank_preference": [1, 2],
        "partner": [16, 17, 18, 19, 20],
        "rank_preference_partner": {
            16: [1, 2],
            17: [1, 2],
            18: [1, 2],
            19: [1],
            20: [1],
        },
    },
    {
        "id": 17,
        "template": "I hate {entity2}. I love {entity1}. {entity1} has a blue shirt.",
        "sentiments": [6.0, 2.0],
        "bias": "proximity",
        "rank_preference": [1, 2],
        "partner": [16, 17, 18, 19, 20],
    },
    {
        "id": 18,
        "template": "I hate {entity2}. {entity1} has a blue shirt. I love {entity1}.",
        "sentiments": [6.0, 2.0],
        "bias": "proximity",
        "rank_preference": [1, 2],
        "partner": [16, 17, 18, 19, 20],
    },
    {
        "id": 19,
        "template": "{entity1} has a blue shirt. I love {entity1}.",
        "sentiments": [6.0],
        "bias": "proximity",
        "rank_preference": [1],
        "partner": [16, 17, 18, 19, 20],
    },
    {
        "id": 20,
        "template": "I love {entity1}. {entity1} has a blue shirt.",
        "sentiments": [4.0],
        "bias": "proximity",
        "rank_preference": [1],
        "partner": [16, 17, 18, 19, 20],
    },
    {
        "id": 21,
        "template": "I root for {entity1}.",
        "sentiments": [6.0],
        "bias": "grammatical",
        "rank_preference": [1],
        "partner": [],
    },
    {
        "id": 22,
        "template": "It's not that I don't like {entity1}, but I prefer {entity2}.",
        "sentiments": [4.0, 6.0],
        "bias": "grammatical",
        "rank_preference": [2, 1],
        "partner": [22, 23, 24],
        "rank_preference_partner": {22: [3, 1], 23: [4, 2], 24: [4, 2]},
    },
    {
        "id": 23,
        "template": "It's not that I hate {entity1}, but I prefer {entity2}.",
        "sentiments": [4.0, 6.0],
        "bias": "grammatical",
        "rank_preference": [2, 1],
    },
    {
        "id": 24,
        "template": "It's not that I hate {entity1}, but I think {entity2} is better.",
        "sentiments": [4.0, 6.0],
        "bias": "grammatical",
        "rank_preference": [2, 1],
    },
    {
        "id": 25,
        "template": "{entity1} is great, but {entity2} is better.",
        "sentiments": [6.0, 6.0],
        "bias": "grammatical",
        "rank_preference": [2, 1],
        "partner": [25, 26],
        "rank_preference_partner": {25: [2, 1], 26: [3, 4], 27: [2, 1]},
    },
    {
        "id": 26,
        "template": "{entity1} is bad, but {entity2} is worse.",
        "sentiments": [2.0, 2.0],
        "bias": "grammatical",
        "rank_preference": [1, 2],
    },
    {
        "id": 27,
        "template": "{entity1} is great, but I prefer {entity2}.",
        "sentiments": [6.0, 6.0],
        "bias": "grammatical",
        "rank_preference": [2, 1],
    },
    {
        "id": 28,
        "template": "I would like {entity1} if {pron1} was more like {entity2}.",
        "sentiments": [4.0, 4.0],
        "bias": "grammatical",
        "rank_preference": [2, 1],
        "partner": [28, 29, 30],
        "rank_preference_partner": {28: [4, 1], 29: [2, 1], 30: [3, 3]},
    },
    {
        "id": 29,
        "template": "I would like {entity1} more if {pron1} was more like {entity2}.",
        "sentiments": [4.0, 4.0],
        "bias": "grammatical",
        "rank_preference": [2, 1],
    },
    {
        "id": 30,
        "template": "I wouldn't like {entity1} more if {pron1} was more like {entity2}.",
        "sentiments": [4.0, 4.0],
        "bias": "grammatical",
        "rank_preference": [1, 1],
    },
    {
        "id": 31,
        "template": "I hate {entity1}, but I would even more if {pron1} was like {entity2}.",
        "sentiments": [2.0, 2.0],
        "bias": "grammatical",
        "rank_preference": [1, 2],
        "partner": [31, 32],
        "rank_preference_partner": {31: [3, 4], 32: [2, 1]},
    },
    {
        "id": 32,
        "template": "I love {entity1}, but I would even more if {pron1} was like {entity2}.",
        "sentiments": [6.0, 6.0],
        "bias": "grammatical",
        "rank_preference": [2, 1],
    },
]


TEMPLATES_BIAS_MENTIONS_NEWSMTSC = [
    {"prompt": "{aspect} is <mask>.", "labels": ["bad", "ok", "good"]},
    {"prompt": "I <mask> {aspect}.", "labels": ["hate", "dislike", "love"]},
    {"prompt": "I think {aspect} is <mask>.", "labels": ["bad", "ok", "good"]},
]

TEMPLATES_BIAS_MENTIONS_OBJECT = [
    {"prompt": "I think the {aspect} was <mask>.", "labels": ["bad", "ok", "good"]},
    {"prompt": "I <mask> the {aspect}.", "labels": ["hate", "like", "love"]},
    {"prompt": "The {aspect} is <mask>.", "labels": ["bad", "ok", "good"]},
]
