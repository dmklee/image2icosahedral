const title = 'Image to Icosahedral Projection for $\\mathrm{SO}(3)$ Object Reasoning from Single-View Images'
const authors = [
	{'name' : 'David M. Klee', link : 'https://dmklee.github.io'},
   	{'name' : 'Ondrej Biza', link : 'https://sites.google.com/view/obiza'},
	{'name' : 'Robert Platt', link : 'https://www.khoury.northeastern.edu/people/robert-platt/'},
   	{'name' : 'Robin Walters', link : 'https://www.khoury.northeastern.edu/people/robin-walters/'},
]
const associations = [
	{'name' : 'Khoury College at Northeastern University',
	 'link' : 'https://www.khoury.northeastern.edu/',
	 'logo' : 'assets/khoury_logo.png',
	},
]
const abstract_text = 'Reasoning about 3D objects based on 2D images is challenging due to variations in appearance caused by viewing the object from different orientations. Tasks such as object classification are invariant to 3D rotations and other such as pose estimation are equivariant. However, imposing equivariance as a model constraint is typically not possible with 2D image input because we do not have an a priori model of how the image changes under out-of-plane object rotations. The only $\\mathrm{SO}(3)$-equivariant models that currently exist require point cloud or voxel input rather than 2D images. In this paper, we propose a novel architecture based on icosahedral group convolutions that reasons in $\\mathrm{SO(3)}$ by learning a projection of the input image onto an icosahedron. The resulting model is approximately equivariant to rotation in $\\mathrm{SO}(3)$. We apply this model to object pose estimation and shape classification tasks and find that it outperforms reasonable baselines.'


function make_header(name) {
	body.append('div')
		.style('margin', '30px 0 10px 0')
		.style('padding-left', '8px')
		.style('padding-bottom', '4px')
		.style('border-bottom', '1px #555 solid')
		.style('width', '100%')
		.append('p')
		.style('font-size', '1.5rem')
		.style('font-style', 'italic')
		.style('margin', '2px 4px')
		.text(name)
}

const max_width = '800px';

var body = d3.select('body')
			 .style('max-width', max_width)
			 .style('margin', '60px auto')
			 .style('margin-top', '100px')
			 .style("font-family", "Garamond")
			 .style("font-size", "1.2rem")

// title
body.append('p')
	.style('font-size', '1.8rem')
	.style('font-weight', 500)
	.style('text-align', 'center')
	.style('margin', '20px auto')
	.text(title)

// authors
var authors_div = body.append('div').attr('class', 'flex-row')
for (let i=0; i < authors.length; i++) {
	authors_div.append('a')
				.attr('href', authors[i]['link'])
				.text(authors[i]['name'])
				.style('margin', '10px')
}

// associations
var associations_div = body.append('div').attr('class', 'flex-row')
for (let i=0; i < associations.length; i++) {
	associations_div.append('a')
					.attr('href', associations[i]['link'])
					.append('img')
					.attr('src', associations[i]['logo'])
					.style('height', '70px')
}


var fig_div = body.append('div')
	.attr('class', 'flex-row')
fig_div
	.append('img')
	.style('margin', 'auto 0')
	.attr('src', 'assets/figure1.png')
	.attr('width', '600px')


// abstract
body.append('div')
	.style('width', '80%')
	.style('margin', '10px auto')
	.style('text-align', 'justify')
	.style('line-height', 1.3)
	.style('font-size', '1rem')
	.append('span').style('font-weight', 'bold').text('Abstract: ')
	.append('span').style('font-weight', 'normal')
	.text(abstract_text)

make_header('Paper')
body.append('div').style('line-height', 1.4).style('font-weight', 'bold').style('font-size', '0.9rem').text(title)
	.append('div').style('font-weight', 'normal').text(authors.map(d => ' '+d.name))
	.append('div').style('font-style', 'italic').text("PMLR Volume on Symmetry and Geometry in Neural Representations")
	.append('div').style('font-style', 'normal').append('a').attr('href', 'https://arxiv.org/abs/2207.08925').text('[Arxiv]')
	
make_header('Poster')
var fig_div = body.append('div')
	.attr('class', 'flex-row')
fig_div
	.append('a')
	.attr('href', 'assets/NeurReps_Poster.pdf')
	.append('img')
	.style('margin', 'auto 0')
	.attr('src', 'assets/poster_preview.png')
	.attr('height', '400px')


make_header('Code')
body.append('div')
	.text('View the code on Github ')
	.append('a')
	.attr('href', 'https://github.com/dmklee/image2icosahedral')
	.text('here.')

make_header('Citation')
body.append('div')
	.append('p')
	.style('border-radius', '6px')
	.style('padding', '10px')
	.style('background-color', '#eee')
	.append('pre')
	.style('font-size', '0.8rem')
	.style('line-height', '1.6')
	.text(`@misc{imagetoico2022,
  title = {I2I: Image to Icosahedral Projection for $\mathrm{SO}(3)$ Object Reasoning from Single-View Images},
  author = {Klee, David and Biza, Ondrej and Platt, Robert and Walters, Robin},
  journal = {arXiv preprint arXiv:2207.08925},
  year = {2022},
}`)

// common syntax
body.selectAll('.flex-row')
	.style('margin', '20px auto')
    .style('display', 'flex')
    .style('justify-content', 'center')
    .style('flex-direction', 'row')
    .style('width', '100%')
body.selectAll('a').style('color', 'blue')
body.selectAll('.content')
	.style('margin', '20px auto')
	.style('width', '90%')
