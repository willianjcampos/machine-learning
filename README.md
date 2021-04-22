<h1>Machine Learning e Data Science em Python</h1>

<br />
<ol>
  <li><a href="#sobre">Sobre os Diretórios e Estrutura</a></li>
  <li><a href="#intro">Introdução</a></li>
  <li><a href="#ic-ml-ds">Inteligência Computacional, <em>Machine Learning</em> e <em>Data Science</em></a></li>
  <li><a href="#definicoes">Definições e Terminologia</a></li>
  <li><a href="#tipos-atributos">Tipos de Atributos</a></li>
  <li><a href="#preditivo-descritivo">Métodos Preditivos e Descritivos</a></li>
  <li><a href="#etapas-ml">Etapas de <em>Machine Learning</em></a></li>
  <li><a href="#tipos-ml">Tipos de Aprendizagem de Máquina</a></li>
  <li><a href="#referencias">Referências</a></li>
</ol>




<br />
<h2 name="sobre">1. Sobre os Diretórios e Estrutura</h2>
<p align="justify">Os arquivos aqui presentes são resultados de estudos realizados através do curso <strong>Machine Learning e Data Science com Python</strong>, ministrado pelo Professor Dr. <strong>Jones Granatyr</strong> pela plataforma <strong>Udemy</strong>.</p>
<p align="justify">Além disso, pelo carater <strong>acadêmico</strong>, em cada um dos diretórios, com exceção do diretório <strong>arquivos</strong>, será dada uma pequena introdução com algumas informações acerca do assunto abordado e referências para abordagens mais detalhadas.</p>
<p align="justify">Abaixo encontram-se os principais diretórios e seus respectivos assuntos.</p>


<h3>1.1. arquivos</h3>
<p align="justify">Nesta diretório, encontram-se as <strong><em>bases de dados</em></strong> utilizadas no decorrer dos estudos. Estes arquivos foram adquiridos através do site da <a href="https://archive.ics.uci.edu/ml/index.php">UCI Machine Learning Repository</a>.</p>


<h3>1.2. a_pre-processamento</h3>
<p align="justify">Nesta diretório, encontram-se os arquivos relacionados ao estudo acerca do <strong>Pré-Processamento de Dados</strong>, dados estes localizados no diretório <strong>arquivos</strong>. Além disso, neste diretório (<em>a_pre-processamento</em>) encontram-se mais informações sobre o assunto e referências para estudos mais detalhados.</p>


<h3>1.3. b_classification</h3>
<p align="justify">Nesta diretório, encontram-se os arquivos relacionados ao estudo de <strong>Classificação de Dados</strong>, dados estes localizados no diretório <strong>arquivos</strong>. Além disso, neste diretório (<em>b_classification</em>) encontram-se mais informações sobre o assunto e referências para estudos mais detalhados.</p>




<br />
<h2 name="intro">2. Introdução</h2>
<p align="justify">De forma geral, <strong>problemas computacionais</strong> são resolvidos por meio da <strong>escrita</strong> de um <strong>programa</strong> que especifica <strong>passo a passo</strong> como o problema deve ser <strong>resolvido</strong>. Podemos definir um <strong>programa</strong> como uma <strong>sequência</strong> de <strong>instruções</strong> que deve ser realizada para <strong>transformar</strong> uma <em>entrada</em> ou um <em>conjunto de entradas</em> em uma <em>saída</em>.</p>

<p align="justify">Porém, algumas <strong>tarefas</strong> do dia-a-dia, que são consideradas <strong>simples</strong>, em nível <strong>computacional</strong> torna-se <strong>complexo</strong> o desenvolvimento de <strong>programas</strong>. Podemos citar como exemplo problemas relacionados ao <strong>reconhecimento de pessoas</strong> através <strong>rosto</strong> ou da <strong>fala</strong>. Que <strong>características</strong> dos rostos ou da fala serão <strong>consideradas</strong>? O que fazer para diferentes <strong>expressões faciais</strong> de uma <em>mesma pessoa</em>? E quando há <strong>alterações</strong> como o uso de <em>óculos</em> ou <em>bigode</em>, <em>cortes de cabelo</em>, <em>mudanças na voz</em> por gripe ou estado de espírito?</p>

<p align="justify">Nós, seres humanos, fazemos este reconhecimento por meio do <strong>reconhecimento de padrões</strong>, onde <strong>aprendemos</strong> o que devemos observar em um <strong>rosto</strong> ou na <strong>fala</strong> para conseguir <strong>identificar</strong> pessoas, e para isso, necessitamos de vários <strong>exemplos</strong> do <strong>rosto</strong> e/ou <strong>fala</strong> com uma <strong>identificação clara</strong>.</p>

<p align="justify">Além do problema relacionado ao reconhecimento de pessoas, podemos levar em consideração que, um bom <strong>médico</strong> consegue, dado o <strong>conjunto de sintomas</strong> e do <strong>resultado</strong> de determinados <strong>exames clínicos</strong>, consegue <strong>diagnosticar</strong> se um paciente está com problemas de saúde. Para tal, o médico utiliza o <strong>conhecimento</strong> adquirido durante sua <strong>formação</strong> e <strong>experiência</strong> proveniente do exercício da <strong>profissão</strong>. Levando estas informações em consideração, como <strong>desenvolver</strong> um <strong>programa</strong> que, dado um <strong>conjunto de sintomas</strong> e os <strong>resultados</strong> dos <strong>exames clínicos</strong>, apresente um <strong>diagnóstico</strong> que seja tão <strong>bom</strong> e <strong>preciso</strong> quanto o de um <strong>médico experiente</strong>?</p>

<p align="justify">Como <strong>desenvolver</strong> um <strong>programa</strong> que <strong>analisa</strong> os dados de <strong>venda</strong> de uma loja para <strong>descobrir</strong> quantas pessoas fizeram mais de uma compra no ano anterior? Podemos utilizar os chamados <a href="https://dicasdeprogramacao.com.br/o-que-e-um-sgbd/">Sistemas de Gerenciamento de Bancos de Dados</a>. Mas e para problemas <strong>mais complexos</strong>, como <strong>identificar</strong> um <strong>conjunto de produtos</strong> que são frequentemente <strong>vendidos em conjunto</strong>, ou <strong>recomendar</strong> novos <strong>produtos</strong> a <strong>clientes</strong> que costumam comprar <strong>produtos semelhantes</strong>, ou ainda <strong>agrupar</strong> os <strong>clientes</strong> ou <strong>consumidores</strong> dos produtos de uma determinada loja em <strong>grupos</strong> para <strong>melhorar</strong> os resultados nas operações de <strong><em>marketing</em></strong>?</p>

<p align="justify">O número de <strong>tarefas complexas</strong> como essas que precisam ser realizadas <strong>diariamente</strong> é <strong>grande</strong>. Além disso, o <strong>volume de informações</strong> que precisam ser consideradas torna <strong>difícil</strong> ou mesmo <strong>impossível</strong> a sua realização por seres humanos. Como resultado, técnicas relacionadas a <strong>Inteligência Computacional</strong>, em particular de <strong><em>Machine Learning</em></strong> (<strong>Aprendizado de Máquina</strong>), têm sido utilizadas com <strong>sucesso</strong> em um grande número de <strong>problemas reais</strong>, incluindo os citados.</p>






<br />
<h2 name="ic-ml-ds">3. Inteligência Computacional, <em>Machine Learning</em> e <em>Data Science</em></h2>
<p align="justify">A <strong>Inteligência Computacional</strong> ou <strong>Inteligência Artificial</strong> (IA) é um campo pertencente a <em>ciência</em> e da <em>engenharia</em> que surgiu após a <strong>Segunda Guerra Mundial </strong>e é uma <strong>ramificação</strong> da <strong>Ciência da Computação</strong>.</p>

<p align="justify">O termo <strong>Inteligência</strong> pode ser definido como a capacidade mental de <em>raciocinar</em>, <em>planejar</em>, <em>resolver problemas</em> e <em>aprender</em>, ao passo de que <strong>Inteligência Computacional</strong> é o ramo da Ciência da Computação que lida com a automação do pensamento e comportamento inteligente.</p>

<p align="justify">Um dos primeiros métodos relacionados a verificação de inteligência em um sistema computacional foi o chamado <strong>Teste de Turing</strong>, conhecido como Jogo da Imitação (<em>Imitation Game</em>). Seu objetivo é <strong>avaliar</strong> se um <em>computador</em> ou <em>programa</em> é <strong>inteligente</strong>. De forma resumida, o teste consiste em um indivíduo (<strong>C</strong>) tenta <strong>distinguir</strong> quem enviou a mensagem: se foi um computador (<strong>A</strong>) ou um ser humano (<strong>B</strong>), conforme mostra a imagem abaixo.</p>

<p align="center"><img src="img/teste-turin.jpg"></p>




<br />
<h2 name="definicoes">4. Definições e Terminologia</h2>
<p align="justify">Parágrafo.</p>




<br />
<h2 name="tipos-atributos">5. Tipos de Atributos</h2>
<p align="justify">Existem <strong>métodos</strong> ou <strong>algoritmos</strong> que usam determinados <strong>tipos</strong> de <strong>variáveis</strong> ou <strong>atributos</strong>. Alguns algoritmos não trabalham com dados numéricos, como por exemplo, os algoritmos de Regras de Associação. Basicamente, existem 2 (dois) tipos principais de <strong>atributos</strong>:</p>
<table border="0">
<tr>
<td width="70%">
<ul>
<li align="justify"><strong>Numéricos</strong>: são representados por dados do tipo numérico, geralmente do tipo int ou float, porém, nem todo número faz parte desta categoria.</li>
<li align="justify"><strong>Categóricos</strong>: são representados pelos demais tipos dados, geralmente do tipo string e expressam categorias ou tipos.</li>
</ul>
</td>
<td><img src="img/atributos.png" alt="Atributo Numérico e Atributo Categórico" width="100%" /></td>
</tr>
</table>


<p align="justify">Os atributos do tipo <strong>NUMÉRICO</strong>, se dividem em outros 2 (dois) tipos:</p>
<table border="0">
<tr>
<td width="70%">
<ul>
<li align="justify"><strong>Contínuo</strong>: representam os dados numéricos reais, ou seja, do tipo float. Podemos citar como exemplo a medição da altura, peso ou temperatura.</li>
<li align="justify"><strong>Discreto</strong>: representam os dados numéricos inteiros, ou seja, do tipo int. Geralmente, estão relacionados a contagem de objetos.</li>
</ul>
</td>
<td><img src="img/atributo-numerico.png" alt="Atributo Numérico" width="100%" /></td>
</tr>
</table>

<p align="justify">Já os atributos do tipo <strong>CATEGÓRICO</strong>, se dividem em outros 2 (dois) tipos:</p>
<table border="0">
<tr>
<td width="70%">
<ul>
<li align="justify"><strong>Nominal</strong>: representam os dados do tipo string que não expressam uma ordem. Podemos citar como exemplo a cor dos olhos, gênero, ID e nome.</li>
<li align="justify"><strong>Ordinal</strong>: representam os dados do tipo string que são categorizados em uma ordem específica. Podemos citar como exemplo os tamanhos P, M e G, onde, P > M > G, ou seja, expressam uma ordem.</li>
</ul>
</td>
<td><img src="img/atributo-categorico.png" alt="Atributo Categórico" width="100%" /></td>
</tr>
</table>

<p align="justify">Assim teremos o seguinte esquema relacionado aos tipos de atributos:</p>
<p align="center"><img src="img/atributos-todos.png" alt="Atributos" /></p>




<br />
<h2 name="preditivo-descritivo">6. Métodos Preditivos e Descritivos</h2>
<p align="justify">Parágrafo.</p>




<br />
<h2 name="etapas-ml">7. Etapas de <em>Machine Learning</em></h2>
<p align="justify">Parágrafo.</p>




<br />
<h2 name="tipos-ml">8. Tipos de Aprendizagem de Máquina</em></h2>
<p align="justify">Parágrafo.</p>




<br />
<h2 name="referencias">9. Referências</h2>
<ul>
  <li align="justify">ALPAYDIN, Ethem. <strong>Introduction to Machine Learning</strong>. 4 ed. Cambridge: MIT, 2020.</li>
  <li align="justify">MOHRI, Mehryar; ROSTAMIZADEH, Afshin; TALWALKAR, Ameet. <strong>Foundations of Machine Learning</strong>. 2 ed. Cambridge: MIT, 2018.</li>
  <li align="justify">RASCHKA, Sebastian; MIRJALILI, Vahid. <strong>Python Machine Learning</strong>: <em>Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2</em>. 3 ed. Mumbai: Packt Publishing, 2019.</li>
</ul>
