import { Component, OnInit } from '@angular/core';
import { NgForm } from '@angular/forms';
import { FileUploadService } from '../services/file-upload.service';
import { SearchService } from '../services/search.service';
@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements OnInit {

  btn1 : any ;
  btn2 : any ;
  //variables to store API response
  returned_img :String = "";
	shortLink: string = "";
	loading: boolean = false; // Flag variable
	file : any;// Variable to store file

  isHidden = false;
  isHidden2 = true;

  
  constructor(private fileUploadService: FileUploadService, private sendImageIdService: SearchService) { }

  ngOnInit(): void {
    this.btn1 =  document.querySelector('.btn1');
    this.btn2  =  document.querySelector('.btn2');
  }

  // On file Select
	onChange(event : any) {
		this.file = event.target.files[0];
	}

  getLink() {
    return this.returned_img.toString();
  }

  myFunction(e:any) {
    var elems = document.querySelectorAll(".active");
    [].forEach.call(elems, function(el:any) {
      el.classList.remove("active");
    });
    e.target.className = "active";
  }

  // OnClick of button Upload
	onUpload() {
		this.loading = !this.loading;
		console.log(this.file);
		this.fileUploadService.upload(this.file).subscribe(
			(event: any) => {
				if (typeof (event) === 'object') {

					// Short link via api response
					this.shortLink = event.link;

					this.loading = false; // Flag variable
				}
			}
		);
	}

  
  onSubmit(f: NgForm ) {
    this.loading = !this.loading;
    this.sendImageIdService.sendId(f.form.value.image_name).subscribe(data => { 
      if (typeof (data) === 'object') {
    this.returned_img = data.img.toString();
    this.loading = false; 
  
  }});

     

  }
  

  onSubmit2(f: NgForm ) {
    console.log(f.form.value)

  }

  makeHide(a : number,evt:any){
    if(a === 1) {
      this.isHidden = true;
      this.isHidden2 = false;
    
      this.btn1.classList.remove('active');
      evt.target.classList.add('active');
    } else {
      this.isHidden = false;
      this.isHidden2 = true;
      this.btn2.classList.remove('active');
      evt.target.classList.add('active');
    }
  }
}



