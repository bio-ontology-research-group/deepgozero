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


#DGL imports
import dgl


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
    relations = {'subclassOf': []}
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
                    relations['subclassOf'].append((c1, c2))
            elif len(it) == 3 and it[1].startswith('ObjectSomeValuesFrom('):
                c1 = to_go(it[0])
                rel = to_rel(it[1])
                c2 = to_go(it[2][:-1])
                if c1 in terms_dict and c2 in terms_dict:
                    c1 = terms_dict[c1]
                    c2 = terms_dict[c2]
                    if rel not in relations:
                        relations[rel] = []
                    relations[rel].append((c1, c2))
        except Exception as e:
            print(f'Ignoring {ax}', e)

    src = []
    dst = []
    etypes = []
    for i, rel in enumerate(relations):
        for s, d in relations[rel]:
            src.append(s)
            dst.append(d)
            etypes.append(i)
    src = th.tensor(src)
    dst = th.tensor(dst)
    etypes = th.tensor(etypes)
    graph = dgl.graph((src, dst), num_nodes=len(terms))
    dgl.save_graphs('data/go.bin', graph, {'etypes': etypes})
    

    df = pd.read_pickle('data/swissprot_interactions.pkl')
    proteins = df['proteins']
    prot_idx = {v: k for k, v in enumerate(proteins)}
    src = []
    dst = []
    for i, row in enumerate(df.itertuples()):
        p_id = prot_idx[row.proteins]
        for p2_id in row.interactions:
            if p2_id in prot_idx:
                p2_id = prot_idx[p2_id]
                src.append(p_id)
                dst.append(p2_id)

    graph = dgl.graph((src, dst), num_nodes=len(proteins))
    graph = dgl.add_self_loop(graph)
    dgl.save_graphs('data/ppi.bin', graph)

    


def to_go(uri):
    return uri[1:-1].replace('http://purl.obolibrary.org/obo/GO_', 'GO:')

def to_rel(uri):
    return uri[len('ObjectSomeValuesFrom(<http://purl.obolibrary.org/obo/'):-1]

if __name__ == '__main__':
    main()
