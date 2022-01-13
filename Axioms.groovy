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
  new File("data/go-plus.owl"))
OWLDataFactory dataFactory = manager.getOWLDataFactory()
ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor()
OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor)
ElkReasonerFactory reasonerFactory = new ElkReasonerFactory()
OWLReasoner reasoner = reasonerFactory.createReasoner(ont, config)
reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)
def renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl()
def shortFormProvider = new SimpleShortFormProvider()

def getName = { cl ->
  def iri = cl.toString()
  def name = iri
  if (iri.startsWith("<http://purl.obolibrary.org/obo/")) {
    name = iri.substring(32, iri.length() - 1)
  } else if (iri.startsWith("<http://aber-owl.net/")) {
    name = iri.substring(21, iri.length() - 1)
  }
  return name
}


def ignoredWords = ["\\r\\n|\\r|\\n", "[()]"]

int dgID = 1;
Map<OWLClassExpression, OWLClass> dgMap = new HashMap<OWLClassExpression, OWLClass>();
ont.getTBoxAxioms().each { axiom ->

    if (axiom.getAxiomType() == AxiomType.SUBCLASS_OF) {
        OWLSubClassOfAxiom subAxiom = (OWLSubClassOfAxiom) axiom;
        OWLClassExpression sub = subAxiom.getSubClass();
        OWLClassExpression sup = subAxiom.getSuperClass();
        OWLClassExpression newSub = null;
        OWLClassExpression newSup = null;
        boolean ok = true;
        if (sub.getClassExpressionType() == ClassExpressionType.valueOf("OBJECT_SOME_VALUES_FROM")) {
            if (!dgMap.containsKey(sub)) {
                IRI iri = IRI.create("http://deepgo.bio2vec.net/DG_" + String.format("%07d", dgID++));
                newSub = dataFactory.getOWLClass(iri);
                OWLEquivalentClassesAxiom newDef =
                    dataFactory.getOWLEquivalentClassesAxiom(newSub, sub);
                manager.addAxiom(ont, newDef);
                dgMap.put(sub, newSub);
            } else {
                newSub = dgMap.get(sub);
            }
            ok = false;
        } else {
            newSub = sub;
        }
        if (sup.getClassExpressionType() == ClassExpressionType.valueOf("OBJECT_SOME_VALUES_FROM")) {
            if (!dgMap.containsKey(sup)) {
                IRI iri = IRI.create("http://deepgo.bio2vec.net/DG_" + String.format("%07d", dgID++));
                newSup = dataFactory.getOWLClass(iri);
                OWLEquivalentClassesAxiom newDef =
                    dataFactory.getOWLEquivalentClassesAxiom(newSup, sup);
                manager.addAxiom(ont, newDef);
                dgMap.put(sup, newSup);
            } else {
                newSup = dgMap.get(sup);
            }
            ok = false;
        } else {
            newSup = sup;
        }

        if (!ok) {
            manager.removeAxiom(ont, axiom);
            OWLSubClassOfAxiom newAxiom = dataFactory.getOWLSubClassOfAxiom(newSub, newSup);
            manager.addAxiom(ont, newAxiom);
        }
    } else if (axiom.getAxiomType() == AxiomType.EQUIVALENT_CLASSES) {
        OWLEquivalentClassesAxiom eqAxiom = (OWLEquivalentClassesAxiom) axiom;
        List<OWLClassExpression> eqAxioms = eqAxiom.getClassExpressionsAsList();
        List<OWLClassExpression> newAxioms = new ArrayList<OWLClassExpression>();
        eqAxioms.each { expr ->
            if (expr.getClassExpressionType() == ClassExpressionType.valueOf("OWL_CLASS")) {
                newAxioms.add(expr);
            } else if (expr.getClassExpressionType() == ClassExpressionType.valueOf("OBJECT_SOME_VALUES_FROM")) {
                OWLClass newExpr = null;
                if (!dgMap.containsKey(expr)) {
                    IRI iri = IRI.create("http://deepgo.bio2vec.net/DG_" + String.format("%07d", dgID++));
                    newExpr = dataFactory.getOWLClass(iri);
                    OWLEquivalentClassesAxiom newDef =
                        dataFactory.getOWLEquivalentClassesAxiom(newExpr, expr);
                    manager.addAxiom(ont, newDef);
                    dgMap.put(expr, newExpr);
                } else {
                    newExpr = dgMap.get(expr);
                }
                newAxioms.add(newExpr);
            } else if (expr.getClassExpressionType() == ClassExpressionType.valueOf("OBJECT_INTERSECTION_OF")) {
                Set<OWLClass> exprSet = new HashSet<OWLClass>();
                expr.asConjunctSet().each { cexpr ->
                    if (cexpr.getClassExpressionType() != ClassExpressionType.valueOf("OWL_CLASS")) {
                        OWLClass newCexpr = null;
                        if (!dgMap.containsKey(cexpr)) {
                            IRI iri = IRI.create("http://deepgo.bio2vec.net/DG_" + String.format("%07d", dgID++));
                            newCexpr = dataFactory.getOWLClass(iri);
                            OWLEquivalentClassesAxiom newDef =
                                dataFactory.getOWLEquivalentClassesAxiom(newCexpr, cexpr);
                            manager.addAxiom(ont, newDef);
                            dgMap.put(cexpr, newCexpr);
                        } else {
                            newCexpr = dgMap.get(cexpr);
                        }
                        exprSet.add(newCexpr);
                    } else {
                        exprSet.add((OWLClass)cexpr);
                    }
                }

                OWLClassExpression newExpr = dataFactory.getOWLObjectIntersectionOf(exprSet);
                newAxioms.add(newExpr);
                
            } else if (expr.getClassExpressionType() == ClassExpressionType.valueOf("OBJECT_UNION_OF")) {
                newAxioms.add(expr);
            }
        }

        OWLEquivalentClassesAxiom newAxiom = dataFactory.getOWLEquivalentClassesAxiom(newAxioms);
        manager.removeAxiom(ont, axiom);
        manager.addAxiom(ont, newAxiom);
        
    } else if (axiom.getAxiomType() == AxiomType.DISJOINT_CLASSES) {
    } else {
    }
}


reasoner = reasonerFactory.createReasoner(ont, config)
reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)

def ec = 0
new InferredEquivalentClassAxiomGenerator().createAxioms(dataFactory, reasoner).each { ax ->
    manager.addAxiom(ont, ax)
    ec += 1
}

println("Inferred Equivalent Classes: " + ec)

def sc = 0
new InferredSubClassAxiomGenerator().createAxioms(dataFactory, reasoner).each { ax ->
    manager.addAxiom(ont, ax)
    sc += 1
}

println("Inferred SubClasses: " + sc)


File newOntFile = new File("data/newGO.owl");
manager.saveOntology(ont, IRI.create(newOntFile.toURI()))

def out = new PrintWriter(
    new BufferedWriter(new FileWriter("data/newgo.sup")))

out.flush()
out.close()
