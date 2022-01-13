@Grapes([
  @Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.2'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.1.0'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.1.0'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.1.0'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-parsers', version='4.1.0'),
  @GrabConfig(systemClassLoader=true)
  ])

import org.semanticweb.owlapi.model.parameters.*
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.*
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.owllink.*;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.search.*;
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.elk.owlapi.ElkReasonerConfiguration
import org.semanticweb.elk.reasoner.config.*
import org.semanticweb.owlapi.manchestersyntax.renderer.*
import org.semanticweb.owlapi.formats.*
import java.util.*


OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ont = manager.loadOntologyFromOntologyDocument(
  new File("data/newGO.owl"))
OWLDataFactory dataFactory = manager.getOWLDataFactory()
ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor()
OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor)
ElkReasonerFactory reasonerFactory = new ElkReasonerFactory()
OWLReasoner reasoner = reasonerFactory.createReasoner(ont, config)
reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)

def renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl()
def shortFormProvider = new SimpleShortFormProvider()

def getName = { cl ->
  return shortFormProvider.getShortForm(cl);
}

def termsDict = new HashSet<String>();
new File("data/terms.csv").each() { line ->
    termsDict.add(line.replaceAll(":", "_"))
}

def out = new PrintWriter(
    new BufferedWriter(new FileWriter("data/newgo.sup")))

def progress = 0;
ont.getClassesInSignature(true).each { cl ->
    def clName = getName(cl);
    if (termsDict.contains(clName)) {
        def superGos = new HashSet<OWLClass>();
        reasoner.getSuperClasses(cl, false).each { node ->
            superGos.addAll(node.getEntities());
        }
        
        out.print(clName.replaceAll("_", ":"));
        for (OWLClass go: superGos) {
            def name = getName(go);
            if (termsDict.contains(name)) {
                out.print("\t" + name.replaceAll("_", ":"));
            }
        }
        out.println();
        progress++;
        if (progress % 1000 == 0) {
            System.out.println("Progress: " + progress);
        }
    }
}

out.flush()
out.close()
