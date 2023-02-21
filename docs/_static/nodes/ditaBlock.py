import os
from docutils import nodes, utils, io
from docutils.parsers.rst import directives
from docutils.utils.error_reporting import ErrorString
from sphinx.util.docutils import SphinxDirective
from bs4 import BeautifulSoup, Tag, Doctype

class dita_block(nodes.literal_block):
    pass

class DitaBlock(SphinxDirective):

    has_content = True
    option_spec = {'file': directives.path}

    def getAllTagNames(self,soup):
        tagNames = {}
        tags = soup.find_all()
        for tag in tags:
            tagNames[tag.name] = tag

    def ditaWalker(self,soup,parent):
        if soup.name is not None:
            for child in soup.children:
                node = None
                getAttrs = True
                if child.name is not None:
                    if child.name == "topic":
                        node = nodes.container()
                    elif child.name == "title":
                        if child.parent.name == "topic":
                            pass
                        else:
                            node = nodes.title()
                    elif child.name == "body":
                        node = nodes.container()
                    elif child.name == "table":
                        node = nodes.table()
                        node['classes'].append("colwidths-given")
                    elif child.name == "tgroup":
                        node = nodes.tgroup()
                        colspecs = child.find_all('colspec',recursive=False)
                        for colspec in colspecs:
                            newSpec = nodes.colspec()
                            if 'colwidth' in colspec.attrs:
                                colw = colspec.attrs['colwidth']
                                if '*' in colw:
                                    colw = int(colw.replace('*',''))
                                else:
                                    colw = int(colw)
                            newSpec.attributes['colwidth'] = colw
                            node += newSpec
                    elif child.name == "colspec":
                        pass
                    elif child.name == "thead":
                        node = nodes.thead()
                    elif child.name == "row":
                        node = nodes.row()
                    elif child.name == "entry":
                        node = nodes.entry()
                    elif child.name == "tbody":
                        node = nodes.tbody()
                    elif child.name == "codeph":
                        node = nodes.literal()
                    elif child.name == "p":
                        node = nodes.paragraph()
                    elif child.name == "ph":
                        node = nodes.strong()
                        #if 'conref' in child.attrs:
                        #    node.attributes['refuri']=child.attrs['conref']
                        #node.attributes['internal']=True
                    elif child.name == "b":
                        node = nodes.strong()
                    elif child.name == "ul":
                        node = nodes.bullet_list()
                    elif child.name == "li":
                        node = nodes.list_item()
                    elif child.name == "xref":
                        node = nodes.reference()
                        if 'scope' in child.attrs:
                            if child.attrs['scope'] != 'external':
                                node.attributes['internal']=True
                        node.attributes['refuri']=child.attrs['href']
                    elif child.name == "codeblock":
                        node = nodes.literal_block()
                        if 'outputclass' in child.attrs:
                            languageBits = child.attrs['outputclass'].split('-')
                            if len(languageBits) == 2:
                                if languageBits[0] == 'language':
                                    node.attributes['language']=languageBits[1]
                    elif child.name == "ol":
                        node = nodes.enumerated_list()
                    try:
                        if getAttrs:
                            for key,value in child.attrs.items():
                                node.attributes[key] = value
                    except:
                        pass
                else:
                    if isinstance(parent, nodes.literal_block):
                        text = child
                    elif isinstance(parent, nodes.literal):
                        text = child.strip()
                    else:
                        if isinstance(child, Doctype) or child.startswith("xml version"):
                            text = ""
                        else:
                            if child.strip() == "":
                                text = ""
                            else:
                                text = child
                    node = nodes.Text(text)
                if node:
                    self.ditaWalker(child,node)
                    parent += node

    def run(self):
        if self.content:
            if 'file' in self.options:
                raise self.error(
                    '"%s" directive may not both specify an external file '
                    'and have content.' % self.name)
            dita = '\n'.join(self.content)
        elif 'file' in self.options:
            source_dir = os.path.dirname(
                os.path.abspath(self.state.document.current_source))
            encoding = self.options.get(
            'encoding', self.state.document.settings.input_encoding)
            path = os.path.normpath(os.path.join(source_dir,
                                                 self.options['file']))
            path = utils.relative_path(None, path)
            try:
                raw_file = io.FileInput(source_path=path,
                                        encoding=encoding)
#                                    error_handler=e_handler)
                # TODO: currently, raw input files are recorded as
                # dependencies even if not used for the chosen output format.
                self.state.document.settings.record_dependencies.add(path)
            except UnicodeEncodeError as error:
                raise self.severe(u'Problems with "%s" directive path:\n%s.'
                                  % (self.name, ErrorString(error)))
            try:
                dita = raw_file.read()
            except IOError as error:
                raise self.severe(u'Problem with "%s" directive:\n%s'
                    % (self.name, ErrorString(error)))
            #attributes['source'] = path
        node = nodes.container()
        node['classes'].append("dita")
        if dita.startswith("<?xml version"):
            dita = dita.split('\n', 1)[-1]
        soup = BeautifulSoup(dita,features="html.parser")
        #self.getAllTagNames(soup)
        self.ditaWalker(soup, node)
        #node = dita_block(dita,dita)
        return [node]