<p align="center">
  <img src="img/Machine-Learning.png" alt="Machine Learning e Data Science em Python" />
</p>
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

<p align="center"><img src="img/teste-de-turing.jpg" alt="Teste de Turing" width="300" /></p>





<br />
<h3>Inteligência Computacional</h3>
<p align="justify">A Inteligência Computacional, também conhecida como Inteligência Artificial, engloba várias subáreas, como a Robótica, Sistemas Multiagentes, Processamento de Linguagem Natural etc. Ao mesmo tempo que são áreas distintas, as mesmas são correlatas e se baseiam em conhecimentos e conceitos de diversas áreas, como Psicologia, Filosofia, Biologia, Estatística dentre outras.</p>

<p align="center"><img src="img/areas-inteligencia-computacional.png" alt="Áreas da Inteligência Computacional / Inteligência Artificial" /></p>

<p align="justify">Muitos pesquisadores optam por utilizar o termo Inteligência Computacional pelo simples fato de que, o termo Inteligência Artificial não possui referências no que diz respeito ao uso do computador. Além disso, algumas revistas científicas utilizam do termo Inteligência Computacional para distinguir determinados tipos ou modalidades de trabalhos e pesquisas científicas na área.</p>




<br />
<h3>Machine Learning</h3>
<p align="justify">Subárea da Inteligência Computacional que proporciona métodos de análise de dados com o intuito de automatizar a construção de modelos analíticos aos sistemas computacionais. Possui a habilidade de aprender e melhorar automaticamente a partir da experiência (E) para esmerar o desempenho (P) ou realizar previsões eficientes e precisas</p>

<p align="justify">Machine Learning ou Aprendizado de Máquina pode ser definido como um método computacional que utiliza experiência (E) para aumentar a performance ou desempenho (P), onde podemos definir como experiência (E) informações anteriores disponibilizadas ao algoritmo (learner), que normalmente assume a forma de dados eletrônicos coletados e disponibilizados para análise.</p>

<p align="justify">Tais dados podem estar na forma de conjuntos de treinamento com identificação humana (supervisionado) ou outros tipos de informações obtidas por meio da interação com o ambiente (não supervisionado). Além disso, sua qualidade e tamanho são cruciais para as realizações das tarefas.</p>

<p align="justify">Em outras palavras, os resultados de um algoritmo de Machine Learning dependem da qualidade e quantidade de dados utilizados. As técnicas de Machine Learning são baseadas em dados que combinam conceitos de ciência da computação com estatística, probabilidade e otimização, das quais, relacionam-se com a análise de dados. Necessita de dados históricos para a criação das relações.</p>

<p align="center"><img src="img/inteligencia-computacional-machine-learning-deep-learning.png" alt="Inteligência Artificial, Machine Learning e Data Science" /></p>



<br />
<h3>Data Science</h3>
<p align="justify">Data Science ou Ciência de Dados é uma área interdisciplinar voltada para o estudo e análise de dados. Sejam dados econômicos, financeiros e sociais, estruturados ou não-estruturados. Visa a extração de conhecimento, detecção de padrões e/ou obtenção de insights para possíveis tomadas de decisão.</p>

<p align="center"><img src="img/inteligencia-computacional-machine-learning-deep-learning2.png" alt="Inteligência Artificial, Machine Learning e Data Science" /></p>




<br />
<h3>Aplicações</h3>
<p align="justify">Existem várias aplicações relacionadas a Machine Learning e Data Science, podemos citar, por exemplo:</p>

<ul>
  <li align="justify">Detecção e Reconhecimento Facial: utilizado, por exemplo, em Sistemas de Segurança. Aplicativo do Google Fotos para agrupar fotografias por faces e facilitar o compartilhamento das fotos.</li>
  <li align="justify">Entretenimento: Jogos como o Kinect da Microsoft para a captura e sincronização dos movimentos com o que é apresentado na tela. Óculos de Realidade Aumentada. (News) IA recria fases de Super Mario.</li>
  <li align="justify">Robótica: Robôs humanoides do Google que aprendem a andar e pegar objetos. Carros Autônomos.</li>
  <li align="justify">Saúde: Medicamentos, cura para doenças, detecção de doenças ou predição de doenças. (News) IA detecta Alzheimer 10 anos antes dos primeiros sintomas.</li>
  <li align="justify">Sistemas de Recomendação: Utilizados pela Netflix, Spotfy e sites de encontro como o eHarmony; Sites de compra para a recomendação de produtos (Amazon).</li>
  <li align="justify">Anúncios personalizados: Utiliza dados para exibir anúncios de acordo com as preferências do usuário. Recomendação de Produtos ou Pessoas. Facebook Ads.</li>
  <li align="justify">Sistemas de Busca: Relaciona as palavras buscadas com os assuntos. Utilizado, por exemplo, pela Google.</li>
  <li align="justify">Artes: Criação de Pinturas e Poemas.</li>
  <li align="justify">Detecção de Malware: (News) Google usa Inteligência Artificial para detectar vírus na PlayStore.</li>
</ul>


<br />
<h2 name="definicoes">4. Definições e Terminologia</h2>
<p align="justify">Como já foi visto anteriormente, existem várias áreas relacionadas a Inteligência Computacional. Abaixo, citaremos apenas algumas áreas e termos utilizados.</p>

<ul>
  <li align="justify">Inteligência Artificial (IA) / Artificial Intelligence (AI): Área da Ciência da Computação responsável pelo desenvolvimento de sistemas que simulem a capacidade humana de resolver problemas. Englobam todas as tecnologias e é o termo mais geral.</li>
  <li align="justify">Inteligência Computacional / Computational Intelligence: Candidato para substituir o termo IA, pois este termo (IA) não diz nada referindo a computadores. Geralmente, envolve Redes Neurais, Computação Evolucionária (Algoritmos Genéticos) e Lógica Fuzzy (Nebulosa). Além disso, para trabalhos na IEEE são apenas estes temas que são aceitos como sendo parte da Inteligência Computacional.</li>
  <li align="justify">Sistemas Especialistas / Expert System: Um dos primeiros esforços da IA em criar sistemas que tem como base o conhecimento de um especialista humano, ou seja, seu objetivo é construir sistemas com base no conhecimento de pessoas que são especialistas na área. Para isso, utiliza-se o conhecimento destes especialistas e insere no sistema.</li>
  <li align="justify">Visão Computacional / Computer Vision: Simula o sentido da visão humana, sendo capaz de detectar ou identificar pessoas, animais e/ou objetos. Estes sistemas geralmente são usados na área de segurança e robótica.</li>
  <li align="justify">Aprendizagem de Máquina / Machine Learning: Baseado no aprendizado a partir de uma base de dados.</li>
  <li align="justify">Processamento de Linguagem Natural (PLN) / Natural Language Processing (NLP): Envolve basicamente o computador compreender a semântica da frase, tanto escrita como falada (Text-To-Speak).</li>
  <li align="justify">Algoritmos Genéticos: Conjunto de algoritmos ou técnicas computacionais de otimização meta-heurístico que se baseiam nos princípios evolutivos da Teoria Sintética da Evolução (Neodarwinismo) de Charles Darwin e Gregor Mendel.</li>
  <li align="justify">Sistemas Multiagente: Subárea da Inteligência Artificial Distribuída que estuda agentes autônomos.</li>
  <li align="justify">Mineração de Dados / Data Mining: Extrair informações de uma base de dados, usando técnicas de Machine Learning; Aplicação prática da Machine Learning.</li>
  <li align="justify">Lógica de Fuzzy (Lógica Nebulosa): Usada principalmente com aplicações na área industrial. Seu objetivo é modelar modos de raciocínio aproximados ao invés de precisos.</li>
  <li align="justify">Raciocínio Baseado em Caos: Abordagem que busca resolver novos problemas adaptando soluções utilizadas para resolver problemas anteriores.</li>
  <li align="justify">Redes Neurais Artificiais (RNA): Tipo de Machine Learning que geralmente está relacionado com Deep Learning.</li>
  <li align="justify">Aprendizagem Profunda / Deep Learning: Trabalha exclusivamente com Redes Neurais com várias camadas. Geralmente, utiliza uma grande quantidade de dados, sendo necessário processadores mais potentes.</li>
  <li align="justify">Big Data: Imenso volume de dados utilizados em algoritmos de Machine Learning ou Deep Learning.</li>
  <li align="justify">Ciência de Dados / Data Science: Carreira de Cientista de Dados que está relacionado a Exploração e análise de dados através de Machine Learning. Para tal, utiliza-se de conhecimentos relacionados a Ciência da Computação e a Estatística (Descritiva e Inferencial).</li>
</ul>
  






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
<td><img src="img/atributos.svg" alt="Atributo Numérico e Atributo Categórico" width="100%" /></td>
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
<td><img src="img/atributo-numerico.svg" alt="Atributo Numérico" width="100%" /></td>
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
<td><img src="img/atributo-categorico.svg" alt="Atributo Categórico" width="100%" /></td>
</tr>
</table>

<p align="justify">Assim teremos o seguinte esquema relacionado aos tipos de atributos:</p>
<p align="center"><img src="img/atributos-todos.svg" alt="Atributos" /></p>




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
