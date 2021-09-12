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
from org.semanticweb.elk.owlapi import ElkReasonerFactory;
from org.semanticweb.elk.owlapi import ElkReasonerConfiguration
from org.semanticweb.elk.reasoner.config import *
from org.semanticweb.owlapi.manchestersyntax.renderer import *
from org.semanticweb.owlapi.formats import *
from org.semanticweb.owlapi.model import OWLAxiom
from java.util import HashSet
from java.io import File




@ck.command()
def main():
    ont_manager = OWLManager.createOWLOntologyManager()
    data_factory = ont_manager.getOWLDataFactory()
    ontology = ont_manager.loadOntologyFromOntologyDocument(
            File("data/go-plus.owl"))

    
    


def to_go(uri):
    return uri[1:-1].replace('http://purl.obolibrary.org/obo/GO_', 'GO:')

def to_rel(uri):
    return uri[len('ObjectSomeValuesFrom(<http://purl.obolibrary.org/obo/'):-1]

if __name__ == '__main__':
    main()
