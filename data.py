#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
import torch as th

import jpype
import jpype.imports
import os

jars_dir = "jars/"
jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'

if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(), "-ea",
        "-Djava.class.path=" + jars,
        convertStrings=False)


# OWLAPI imports
from org.semanticweb.owlapi.model import OWLOntology
from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.reasoner import ConsoleProgressMonitor
from org.semanticweb.owlapi.reasoner import SimpleConfiguration
from org.semanticweb.owlapi.reasoner import InferenceType
from org.semanticweb.owlapi.util import InferredClassAssertionAxiomGenerator
from org.semanticweb.owlapi.util import InferredEquivalentClassAxiomGenerator
from org.semanticweb.owlapi.util import InferredSubClassAxiomGenerator
from de.tudresden.inf.lat.jcel.owlapi.main import JcelReasoner
from de.tudresden.inf.lat.jcel.ontology.normalization import OntologyNormalizer
from de.tudresden.inf.lat.jcel.ontology.axiom.extension import IntegerOntologyObjectFactoryImpl
from de.tudresden.inf.lat.jcel.owlapi.translator import ReverseAxiomTranslator
from org.semanticweb.owlapi.model import OWLAxiom
from java.util import HashSet
from java.io import File




@ck.command()
def main():
    ont_manager = OWLManager.createOWLOntologyManager()
    data_factory = ont_manager.getOWLDataFactory()
    ontology = ont_manager.loadOntologyFromOntologyDocument(
            File("data/go.owl"))
    jReasoner = JcelReasoner(ontology, False)
    rootOnt = jReasoner.getRootOntology()
    translator = jReasoner.getTranslator()
    axioms = HashSet()
    axioms.addAll(rootOnt.getAxioms())
    translator.getTranslationRepository().addAxiomEntities(
        rootOnt)
    for ont in rootOnt.getImportsClosure():
        axioms.addAll(ont.getAxioms())
        translator.getTranslationRepository().addAxiomEntities(
            ont)

    intAxioms = translator.translateSA(axioms)

    normalizer = OntologyNormalizer()
    factory = IntegerOntologyObjectFactoryImpl()
    normalizedOntology = normalizer.normalize(intAxioms, factory)
    rTranslator = ReverseAxiomTranslator(translator, ontology)

    gos_df = pd.read_pickle('data/terms.pkl')
    terms = gos_df['gos'].values
    terms_dict = {v:k for k, v in enumerate(terms)}
    relations = []
    rel_id = {}
    subclass = []
    gos = {}
    prefix = '<http://purl.obolibrary.org/obo/GO_'
    lp = len(prefix)
    for ax in normalizedOntology:
        try:
            axiom = f'{rTranslator.visit(ax)}'[11:-1]
            it = axiom.split(' ')
            if len(it) == 2:
                c1 = to_go(it[0])
                c2 = to_go(it[1])
                if c1 in terms_dict and c2 in terms_dict:
                    c1 = terms_dict[c1]
                    c2 = terms_dict[c2]
                    subclass.append((c1, c2))
            elif len(it) == 3 and it[1].startswith('ObjectSomeValuesFrom('):
                c1 = to_go(it[0])
                rel = to_rel(it[1])
                c2 = to_go(it[2][:-1])
                if c1 in terms_dict and c2 in terms_dict:
                    c1 = terms_dict[c1]
                    c2 = terms_dict[c2]
                    if rel not in rel_id:
                        rel_id[rel] = len(rel_id)
                    relations.append((c1, rel_id[rel], c2))
        except Exception as e:
            print(f'Ignoring {ax}', e)

    rel_id['hasFunction'] = len(rel_id)
    df = pd.read_pickle('data/train_data.pkl')
    hasfunc = []
    for i, row in enumerate(df.itertuples()):
        p_id = i
        for go_id in row.exp_annotations:
            if go_id in terms_dict:
                go_id = terms_dict[go_id]
                hasfunc.append((p_id, go_id))
    sub_df = pd.DataFrame({'subclass': subclass})
    rel_df = pd.DataFrame({'relations': relations})
    hf_df = pd.DataFrame({'hasfunc': hasfunc})
    sub_df.to_pickle('data/train_subclass.pkl')
    rel_df.to_pickle('data/train_relations.pkl')
    hf_df.to_pickle('data/train_hasfunc.pkl')

    rel_id_df = pd.DataFrame({'relations': list(rel_id.keys()), 'ids': list(rel_id.values())})
    rel_id_df.to_pickle('data/relations.pkl')
    
    


def to_go(uri):
    return uri[1:-1].replace('http://purl.obolibrary.org/obo/GO_', 'GO:')

def to_rel(uri):
    return uri[len('ObjectSomeValuesFrom(<http://purl.obolibrary.org/obo/'):-1]

if __name__ == '__main__':
    main()
