import { Component, Input, OnInit } from '@angular/core';
import { Form, NgForm } from '@angular/forms';
import { SafeResourceUrl } from '@angular/platform-browser';
import { ParentService } from '../services/parent.service';
import { ProcessFormService } from '../services/process-form.service';
import { ImageResponse } from '../shared/image-response';

@Component({
  selector: 'app-processing',
  templateUrl: './processing.component.html',
  styleUrls: ['./processing.component.css']
})
export class ProcessingComponent implements OnInit {

  processingOptions: Array<any> = [
    { name: 'enhance_large_scale_clouds', value: false },
    { name: 'enhance_small_scale_clouds', value: false },
    { name: 'denoise', value: false },
    { name: 'gama_correction', value: false },
    { name: 'enhance_brightness', value: false },
    { name: 'enhance_contrast', value: false }
  ];

   resp_after_processing : ImageResponse | undefined;
   list : any;
   process_old : SafeResourceUrl | undefined ;
   process_new : SafeResourceUrl | undefined;

   loadingProcess: boolean = false; // Flag variable

  constructor(private sharedService : ParentService, private processingService : ProcessFormService) { 
  }


  
  ngOnInit(): void {
    this.setProcessImgProperties();
  }

  setProcessImgProperties() {

    this.list = this.sharedService.getImagesTrustedLinks();
    this.process_old = this.list[0];
    this.process_new = this.list[1];
  }

  onSubmit() {
    this.loadingProcess = !this.loadingProcess;
   
     this.processingService.regenerate(this.processingOptions).subscribe(data => { 
     if (typeof (data) === 'object') {
     this.resp_after_processing = data;
     this.sharedService.updateUrls(this.resp_after_processing);
     this.loadingProcess = false; 
     this.setProcessImgProperties();
   }});
  }


   
}
